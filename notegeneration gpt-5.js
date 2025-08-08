// noteGeneration.js

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
  // Prepare the messages array for the Responses API
  const messages = [
    { role: "system", content: finalPromptText },
    { role: "user",   content: transcriptionText }
  ];
  // Call the Responses API with GPT-5 and streaming
  const resp = await fetch("https://api.openai.com/v1/responses", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model: "gpt-5",
      input: messages.map(m => ({
        role: m.role,
        content: [{ type: "input_text", text: m.content }]
      })),
      stream: true
    })
  });
    await streamOpenAIResponse(resp, {
      onDelta: (textChunk) => {
        generatedNoteField.value += textChunk;
      },
      onDone: () => {},
      onError: (err) => {
        console.error("Streaming error:", err);
        alert("Error during note generation");
      }
    });

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

async function streamOpenAIResponse(resp, {
  onDelta = () => {},
  onDone = () => {},
  onError = (e) => { console.error(e); },
} = {}) {
  if (!resp.ok || !resp.body) {
    const text = await resp.text().catch(() => "");
    throw new Error(`OpenAI error ${resp.status}: ${text}`);
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop() ?? "";

      for (const part of parts) {
        const lines = part.split("\n");
        let event = null;
        let dataStr = null;
        for (const line of lines) {
          if (line.startsWith("event:")) event = line.slice(6).trim();
          if (line.startsWith("data:"))  dataStr = line.slice(5).trim();
        }
        if (!dataStr) continue;
        if (dataStr === "[DONE]") { onDone(); return; }

        let payload;
        try { payload = JSON.parse(dataStr); } catch { continue; }

        if (payload.type === "response.output_text.delta" && typeof payload.delta === "string") {
          onDelta(payload.delta);
        }
        if (payload.type === "response.completed") {
          onDone(payload);
          return;
        }
        if (payload.type === "response.error") {
          onError(new Error(payload.error?.message || "Unknown streaming error"));
          return;
        }
      }
    }
    onDone();
  } catch (e) {
    onError(e);
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
