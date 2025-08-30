// noteGeneration.js
 // Use the same Vercel proxy as transcription (routes JSON to /v1/chat/completions)
 const PROXY_URL = "https://transcribe-notes-five.vercel.app/api/transcribe";
// Utility function to hash a string (used for storing prompts keyed by API key)
function hashString(str) {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash |= 0; // Convert to 32bit integer
  }
  return hash.toString();
}

// Helper functions for base64 conversions (kept in case they’re used elsewhere)
function arrayBufferToBase64(buffer) {
  let binary = "";
  const bytes = new Uint8Array(buffer);
  for (let i = 0; i < bytes.byteLength; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return window.btoa(binary);
}

function base64ToArrayBuffer(base64) {
  const binary = window.atob(base64);
  const len = binary.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}

// Since encryption is no longer needed, the decryption functions are removed.
// We now assume that the plain API key is stored in sessionStorage under "user_api_key".

// Returns a storage key for a given prompt slot and API key
function getPromptStorageKey(slot) {
  // Now retrieves the key from "user_api_key" directly.
  const apiKey = sessionStorage.getItem("user_api_key") || "";
  const hashedApiKey = hashString(apiKey);
  return "customPrompt_" + hashedApiKey + "_" + slot;
}

// Auto-resizes a textarea based on its content
function autoResize(textarea) {
  textarea.style.height = "auto";
  textarea.style.height = textarea.scrollHeight + "px";
}

// Loads the stored prompt for a given slot into the custom prompt textarea
function loadPromptForSlot(slot) {
  const key = getPromptStorageKey(slot);
  const storedPrompt = localStorage.getItem(key);
  const customPromptTextarea = document.getElementById("customPrompt");
  if (customPromptTextarea) {
    customPromptTextarea.value = storedPrompt ? storedPrompt : "";
    autoResize(customPromptTextarea);
  }
}

// Formats milliseconds into a human-readable string
function formatTime(ms) {
  const totalSec = Math.floor(ms / 1000);
  if (totalSec < 60) {
    return totalSec + " sec";
  } else {
    const minutes = Math.floor(totalSec / 60);
    const seconds = totalSec % 60;
    return minutes + " min" + (seconds > 0 ? " " + seconds + " sec" : "");
  }
}

// Handles the note generation process using the OpenAI API
async function generateNote() {
  const transcriptionElem = document.getElementById("transcription");
  if (!transcriptionElem) {
    alert("No transcription text available.");
    return;
  }
  const transcriptionText = transcriptionElem.value.trim();
  if (!transcriptionText) {
    alert("No transcription text available.");
    return;
  }
  
  const customPromptTextarea = document.getElementById("customPrompt");
  const promptText = customPromptTextarea ? customPromptTextarea.value : "";
  const generatedNoteField = document.getElementById("generatedNote");
  if (!generatedNoteField) return;
  
  // Reset generated note field and start timer
  generatedNoteField.value = "";
  const noteTimerElement = document.getElementById("noteTimer");
  const noteStartTime = Date.now();
  if (noteTimerElement) {
    noteTimerElement.innerText = "Note Generation Timer: 0 sec";
  }
  const noteTimerInterval = setInterval(() => {
    if (noteTimerElement) {
      noteTimerElement.innerText = "Note Generation Timer: " + formatTime(Date.now() - noteStartTime);
    }
  }, 1000);
  
  // Retrieve the plain API key from sessionStorage
  const apiKey = sessionStorage.getItem("user_api_key");
  if (!apiKey) {
    alert("No API key available for note generation.");
    clearInterval(noteTimerInterval);
    return;
  }
  
  // Add the fixed formatting instruction as a hidden prompt component.
  const baseInstruction = `
Do not use bold text. Do not use asterisks (*) or Markdown formatting anywhere in the output.
All headings should be plain text with a colon, like 'Bakgrunn:'.`.trim();

  // Append the hidden instruction to the user's prompt so it is always included.
  const finalPromptText = promptText + "\n\n" + baseInstruction;
  
  try {
    const response = await fetch(PROXY_URL + "?mode=chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": apiKey
      },
      body: JSON.stringify({
        model: "mistral-large-latest",
        messages: [
          { role: "system", content: finalPromptText },
          { role: "user", content: transcriptionText }
        ],
        temperature: 0.2,
        stream: true,
      })
    });
        const reader    = response.body.getReader();
    const decoder   = new TextDecoder("utf-8");
    // buffer for incomplete SSE events
    let   sseBuffer = "";
    let   done      = false;

    while (!done) {
      const { value, done: doneReading } = await reader.read();
      done = doneReading;

      // — start streaming parse —
      // decode with stream:true to preserve partial UTF-8 codepoints
      const chunkText = decoder.decode(value, { stream: true });
      sseBuffer += chunkText;

      // extract all full SSE events terminated by "\n\n"
      let boundary;
      while ((boundary = sseBuffer.indexOf("\n\n")) !== -1) {
        const event = sseBuffer.slice(0, boundary).trim();
        sseBuffer = sseBuffer.slice(boundary + 2);

        if (!event.startsWith("data: ")) continue;
        const payload = event.replace(/^data: /, "");
        if (payload === "[DONE]") { done = true; break; }

        try {
          const parsed    = JSON.parse(payload);
          const textChunk = parsed.choices[0].delta?.content || "";
          generatedNoteField.value += textChunk;
          autoResize(generatedNoteField);
        } catch (err) {
          console.error("Malformed SSE JSON, skipping:", err, payload);
        }
      }
      // — end streaming parse —
    }

    clearInterval(noteTimerInterval);
    if (noteTimerElement) {
      noteTimerElement.innerText = "Text generation completed!";
    }
  } catch (error) {
    clearInterval(noteTimerInterval);
    if (generatedNoteField) {
      generatedNoteField.value = "Error generating note: " + error;
    }
    if (noteTimerElement) {
      noteTimerElement.innerText = "";
    }
  }
}
 
// Initializes note generation functionality, including prompt slot handling and event listeners.
function initNoteGeneration() {
  const promptSlotSelect = document.getElementById("promptSlot");
  const customPromptTextarea = document.getElementById("customPrompt");
  const generateNoteButton = document.getElementById("generateNoteButton");
  if (!promptSlotSelect || !customPromptTextarea || !generateNoteButton) return;
  
  // Disable the Generate Note button if consent is not accepted.
  if (document.cookie.indexOf("user_consent=accepted") === -1) {
    generateNoteButton.disabled = true;
    generateNoteButton.title = "Note generation is disabled until you accept cookies/ads.";
  }
  
  // Load the stored prompt for the current slot.
  loadPromptForSlot(promptSlotSelect.value);
  
  // Save prompt changes on input.
  customPromptTextarea.addEventListener("input", () => {
    const currentSlot = promptSlotSelect.value;
    const key = getPromptStorageKey(currentSlot);
    localStorage.setItem(key, customPromptTextarea.value);
    autoResize(customPromptTextarea);
  });
  
  // Load the prompt when the slot changes.
  promptSlotSelect.addEventListener("change", () => {
    loadPromptForSlot(promptSlotSelect.value);
  });
  
  generateNoteButton.addEventListener("click", generateNote);
}
 
export { initNoteGeneration };
