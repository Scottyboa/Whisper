// recording.js
// Updated recording module without encryption/HMAC mechanisms,
// processing audio chunks using OfflineAudioContext,
// and implementing a client‑side transcription queue that sends each processed chunk directly to OpenAI's Whisper API.

function hashString(str) {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash |= 0; // Convert to 32-bit signed integer
  }
  // Convert to an unsigned 32-bit integer and return as string.
  return (hash >>> 0).toString();
}

const DEBUG = true;
function logDebug(message, ...optionalParams) {
  if (DEBUG) {
    console.debug(new Date().toISOString(), "[DEBUG]", message, ...optionalParams);
  }
}
function logInfo(message, ...optionalParams) {
  console.info(new Date().toISOString(), "[INFO]", message, ...optionalParams);
}
function logError(message, ...optionalParams) {
  console.error(new Date().toISOString(), "[ERROR]", message, ...optionalParams);
}

const MIN_CHUNK_DURATION = 120000; // 120 seconds
const MAX_CHUNK_DURATION = 120000; // 120 seconds
const watchdogThreshold = 1500;   // 1.5 seconds with no frame

let mediaStream = null;
let processedAnyAudioFrames = false;
let audioReader = null;
let recordingStartTime = 0;
// Accumulate time from all active segments
let accumulatedRecordingTime = 0;
let recordingTimerInterval;
let completionTimerInterval = null;
let completionStartTime = 0;
let groupId = null;
let chunkNumber = 1;
let manualStop = false;
let transcriptChunks = {};  // {chunkNumber: transcript}
let pollingIntervals = {};  // (removed polling functions, kept for legacy structure)

let chunkStartTime = 0;
let lastFrameTime = 0;
let chunkTimeoutId;

let chunkProcessingLock = false;
let pendingStop = false;
let finalChunkProcessed = false;
let recordingPaused = false;
let audioFrames = []; // Buffer for audio frames

// --- New Transcription Queue Variables ---
let transcriptionQueue = [];  // Queue of { chunkNumber, blob }
let isProcessingQueue = false;

// --- Utility Functions ---
function updateStatusMessage(message, color = "#333") {
  const statusElem = document.getElementById("statusMessage");
  if (statusElem) {
    statusElem.innerText = message;
    statusElem.style.color = color;
  }
}

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

function updateRecordingTimer() {
  // Timer shows accumulated time plus current active segment time
  const elapsed = accumulatedRecordingTime + (Date.now() - recordingStartTime);
  const timerElem = document.getElementById("recordTimer");
  if (timerElem) {
    timerElem.innerText = "Recording Timer: " + formatTime(elapsed);
  }
}

function stopMicrophone() {
  if (mediaStream) {
    mediaStream.getTracks().forEach(track => track.stop());
    mediaStream = null;
    logInfo("Microphone stopped.");
  }
  if (audioReader) {
    audioReader.cancel();
    audioReader = null;
  }
}

// --- Base64 Helper Functions (kept for legacy) ---
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

// --- Device Token Management ---
function getDeviceToken() {
  let token = localStorage.getItem("device_token");
  if (!token) {
    token = crypto.randomUUID();
    localStorage.setItem("device_token", token);
  }
  return token;
}

// --- API Key Retrieval ---
// With encryption removed, we now simply get the API key from sessionStorage.
function getAPIKey() {
  return sessionStorage.getItem("user_api_key");
}

// --- File Blob Processing ---
// Previously used for encryption; now simply returns the original blob along with markers.
async function encryptFileBlob(blob) {
  const apiKey = getAPIKey();
  if (!apiKey) throw new Error("API key not available");
  const deviceToken = getDeviceToken();
  const apiKeyMarker = hashString(apiKey);
  const deviceMarker = hashString(deviceToken);
  // Return the original blob without any encryption; iv and salt are empty.
  return {
    encryptedBlob: blob,
    iv: "",
    salt: "",
    apiKeyMarker,
    deviceMarker
  };
}

// --- OfflineAudioContext Processing ---
// This function takes interleaved PCM samples (Float32Array), the original sample rate, and the number of channels,
// converts the audio to mono (averaging channels if needed), resamples to 16kHz, and applies 0.3s fade‑in/out.
// It returns a 16-bit PCM WAV Blob.
async function processAudioUsingOfflineContext(pcmFloat32, originalSampleRate, numChannels) {
  const targetSampleRate = 16000;
  
  // Calculate the number of frames
  const numFrames = pcmFloat32.length / numChannels;
  
  // Create an AudioBuffer in a temporary AudioContext
  let tempCtx = new AudioContext();
  let originalBuffer = tempCtx.createBuffer(numChannels, numFrames, originalSampleRate);
  
  if (numChannels === 1) {
    originalBuffer.copyToChannel(pcmFloat32, 0);
  } else {
    // Deinterleave and copy each channel
    for (let ch = 0; ch < numChannels; ch++) {
      let channelData = new Float32Array(numFrames);
      for (let i = 0; i < numFrames; i++) {
        channelData[i] = pcmFloat32[i * numChannels + ch];
      }
      originalBuffer.copyToChannel(channelData, ch);
    }
  }
  // Convert to mono by averaging channels if necessary
  let monoBuffer;
  if (numChannels > 1) {
    let monoData = new Float32Array(numFrames);
    for (let i = 0; i < numFrames; i++) {
      let sum = 0;
      for (let ch = 0; ch < numChannels; ch++) {
        sum += originalBuffer.getChannelData(ch)[i];
      }
      monoData[i] = sum / numChannels;
    }
    monoBuffer = tempCtx.createBuffer(1, numFrames, originalSampleRate);
    monoBuffer.copyToChannel(monoData, 0);
  } else {
    monoBuffer = originalBuffer;
  }
  tempCtx.close();
  
  // Set up OfflineAudioContext for resampling
  const duration = monoBuffer.duration;
  const offlineCtx = new OfflineAudioContext(1, targetSampleRate * duration, targetSampleRate);
  
  const source = offlineCtx.createBufferSource();
  source.buffer = monoBuffer;
  
// Modified code snippet to fix the negative time error:
const gainNode = offlineCtx.createGain();
const fadeDuration = 0.3;
gainNode.gain.setValueAtTime(0, 0);
gainNode.gain.linearRampToValueAtTime(1, fadeDuration);

// Compute fade-out start time, ensuring it's non-negative
const fadeOutStart = Math.max(0, duration - fadeDuration);
if (duration < fadeDuration * 2) {
  console.warn(`[Audio] Short chunk (${duration.toFixed(2)}s) — fade-in/out may be squished`);
}

gainNode.gain.setValueAtTime(1, fadeOutStart);
gainNode.gain.linearRampToValueAtTime(0, duration);
  
  source.connect(gainNode).connect(offlineCtx.destination);
  source.start(0);
  
  const renderedBuffer = await offlineCtx.startRendering();
  const processedData = renderedBuffer.getChannelData(0);
  const processedInt16 = floatTo16BitPCM(processedData);
  const wavBlob = encodeWAV(processedInt16, targetSampleRate, 1);
  return wavBlob;
}

// --- New: Transcribe Chunk Directly ---
// Sends the WAV blob directly to OpenAI's Whisper API and returns the transcript.
async function transcribeChunkDirectly(wavBlob, chunkNum) {
  const apiKey = getAPIKey();
  if (!apiKey) throw new Error("API key not available for transcription");
  
  const formData = new FormData();
  formData.append("file", wavBlob, `chunk_${chunkNum}.wav`);
  formData.append("model", "gpt-4o-transcribe");
  formData.append("temperature", "0.2");
  formData.append("prompt", 
    "Transcribe only spoken words. Exclude non-verbal sounds and background noise. Do NOT omit, summarize, or alter any spoken words or sentences. Do NOT repeat the same sentence multiple times in succession."
  );
  
  try {
    const response = await fetch("https://api.openai.com/v1/audio/transcriptions", {
      method: "POST",
      headers: {
        "Authorization": "Bearer " + apiKey
      },
      body: formData
    });
    if (!response.ok) {
      // Attempt to parse JSON error
      let err;
      try { err = await response.json(); } catch { err = null; }
      // Handle insufficient quota specifically
      if (err?.error?.code === "insufficient_quota") {
        logError("Insufficient quota for transcription:", err);
        updateStatusMessage(
          "Error: You have exceeded your OpenAI quota or have no credits left.",
          "red"
        );
        return "";
      }
      // Fallback for other errors
      const msg = err?.error?.message || await response.text();
      throw new Error(`OpenAI API error: ${msg}`);
    }
    const result = await response.json();
    return result.text || "";
  } catch (error) {
    logError(`Error transcribing chunk ${chunkNum}:`, error);
    // Show tailored message if it really was a quota issue
    updateStatusMessage(
      error.message.includes("insufficient_quota")
        ? "Error: You have exceeded your OpenAI quota or have no credits left."
        : "Error transcribing: please try again",
      "red"
    );
    return `[Error transcribing chunk ${chunkNum}]`;
  }
}

// --- Transcription Queue Processing ---
// Adds a processed chunk to the queue and processes chunks sequentially.
function enqueueTranscription(wavBlob, chunkNum) {
  transcriptionQueue.push({ chunkNum, wavBlob });
  processTranscriptionQueue();
}

async function processTranscriptionQueue() {
  if (isProcessingQueue) return;
  isProcessingQueue = true;
  
  while (transcriptionQueue.length > 0) {
    const { chunkNum, wavBlob } = transcriptionQueue.shift();
    logInfo(`Transcribing chunk ${chunkNum}...`);
    const transcript = await transcribeChunkDirectly(wavBlob, chunkNum);
    transcriptChunks[chunkNum] = transcript;
    updateTranscriptionOutput();
  }
  
  isProcessingQueue = false;
}

// --- Removed: Polling functions (pollChunkTranscript) since we now transcribe directly ---

// --- Audio Chunk Processing ---
async function processAudioChunkInternal(force = false) {
  if (audioFrames.length === 0) {
    logDebug("No audio frames to process.");
    return;
  }
  // Mark that we have processed at least one frame set.
  processedAnyAudioFrames = true;
  
  logInfo(`Processing ${audioFrames.length} audio frames for chunk ${chunkNumber}.`);
  const framesToProcess = audioFrames;
  audioFrames = []; // Clear the buffer
  const sampleRate = framesToProcess[0].sampleRate;
  const numChannels = framesToProcess[0].numberOfChannels;
  let pcmDataArray = [];
  for (const frame of framesToProcess) {
    const numFrames = frame.numberOfFrames;
    if (numChannels === 1) {
      const channelData = new Float32Array(numFrames);
      frame.copyTo(channelData, { planeIndex: 0 });
      pcmDataArray.push(channelData);
    } else {
      let channelData = [];
      for (let c = 0; c < numChannels; c++) {
        const channelArray = new Float32Array(numFrames);
        frame.copyTo(channelArray, { planeIndex: c });
        channelData.push(channelArray);
      }
      const interleaved = new Float32Array(numFrames * numChannels);
      for (let i = 0; i < numFrames; i++) {
        for (let c = 0; c < numChannels; c++) {
          interleaved[i * numChannels + c] = channelData[c][i];
        }
      }
      pcmDataArray.push(interleaved);
    }
    frame.close();
  }
  const totalLength = pcmDataArray.reduce((sum, arr) => sum + arr.length, 0);
  const pcmFloat32 = new Float32Array(totalLength);
  let offset = 0;
  for (const arr of pcmDataArray) {
    pcmFloat32.set(arr, offset);
    offset += arr.length;
  }
  
  // Process the raw audio samples using OfflineAudioContext:
  // Convert to mono, resample to 16kHz, and apply 0.3s fade-in/out.
  const wavBlob = await processAudioUsingOfflineContext(pcmFloat32, sampleRate, numChannels);
  
  // Instead of uploading to a backend, enqueue this processed chunk for direct transcription.
  enqueueTranscription(wavBlob, chunkNumber);
  
  chunkNumber++;
}

async function safeProcessAudioChunk(force = false) {
  if (manualStop && finalChunkProcessed) {
    logDebug("Final chunk already processed; skipping safeProcessAudioChunk.");
    return;
  }
  if (chunkProcessingLock) {
    logDebug("Chunk processing is locked; skipping safeProcessAudioChunk.");
    return;
  }
  chunkProcessingLock = true;
  await processAudioChunkInternal(force);
  chunkProcessingLock = false;
  if (pendingStop) {
    pendingStop = false;
    finalizeStop();
  }
}

function finalizeStop() {
  completionStartTime = Date.now();
  completionTimerInterval = setInterval(() => {
    const timerElem = document.getElementById("transcribeTimer");
    if (timerElem) {
      timerElem.innerText = "Completion Timer: " + formatTime(Date.now() - completionStartTime);
    }
  }, 1000);
  const startButton = document.getElementById("startButton");
  const stopButton = document.getElementById("stopButton");
  const pauseResumeButton = document.getElementById("pauseResumeButton");
  if (startButton) startButton.disabled = false;
  if (stopButton) stopButton.disabled = true;
  if (pauseResumeButton) pauseResumeButton.disabled = true;
  logInfo("Recording stopped by user. Finalizing transcription.");
  // Optionally, you could wait here for the queue to empty before declaring completion.
}

function updateTranscriptionOutput() {
  const sortedKeys = Object.keys(transcriptChunks).map(Number).sort((a, b) => a - b);
  let combinedTranscript = "";
  sortedKeys.forEach(key => {
    combinedTranscript += transcriptChunks[key] + " ";
  });
  const transcriptionElem = document.getElementById("transcription");
  if (transcriptionElem) {
    transcriptionElem.value = combinedTranscript.trim();
  }
  if (manualStop && Object.keys(transcriptChunks).length >= (chunkNumber - 1)) {
    clearInterval(completionTimerInterval);
    updateStatusMessage("Transcription finished!", "green");
    logInfo("Transcription complete.");
  }
}

function floatTo16BitPCM(input) {
  const output = new Int16Array(input.length);
  for (let i = 0; i < input.length; i++) {
    let s = Math.max(-1, Math.min(1, input[i]));
    output[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
  }
  return output;
}

function encodeWAV(samples, sampleRate, numChannels) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  function writeString(offset, string) {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }
  writeString(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * numChannels * 2, true);
  view.setUint16(32, numChannels * 2, true);
  view.setUint16(34, 16, true);
  writeString(36, 'data');
  view.setUint32(40, samples.length * 2, true);
  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    view.setInt16(offset, samples[i], true);
  }
  return new Blob([view], { type: 'audio/wav' });
}

function scheduleChunk() {
  if (manualStop || recordingPaused) {
    logDebug("Scheduler suspended due to manual stop or pause.");
    return;
  }
  const elapsed = Date.now() - chunkStartTime;
  const timeSinceLast = Date.now() - lastFrameTime;
  if (elapsed >= MAX_CHUNK_DURATION || (elapsed >= MIN_CHUNK_DURATION && timeSinceLast >= watchdogThreshold)) {
    logInfo("Scheduling condition met; processing chunk.");
    safeProcessAudioChunk();
    chunkStartTime = Date.now();
    scheduleChunk();
  } else {
    chunkTimeoutId = setTimeout(scheduleChunk, 500);
  }
}

function resetRecordingState() {
  Object.values(pollingIntervals).forEach(interval => clearInterval(interval));
  pollingIntervals = {};
  clearTimeout(chunkTimeoutId);
  clearInterval(recordingTimerInterval);
  transcriptChunks = {};
  audioFrames = [];
  chunkStartTime = Date.now();
  lastFrameTime = Date.now();
  manualStop = false;
  finalChunkProcessed = false;
  recordingPaused = false;
  groupId = Date.now().toString();
  chunkNumber = 1;
  // Reset accumulated recording time for a new session
  accumulatedRecordingTime = 0;
}

function initRecording() {
  const startButton = document.getElementById("startButton");
  const stopButton = document.getElementById("stopButton");
  const pauseResumeButton = document.getElementById("pauseResumeButton");
  if (!startButton || !stopButton || !pauseResumeButton) return;

  startButton.addEventListener("click", async () => {
    // Retrieve the API key before starting.
    const apiKey = getAPIKey();
    if (!apiKey || !apiKey.startsWith("sk-")) {
      alert("Please enter a valid OpenAI API key before starting the recording.");
      return;
    }
    resetRecordingState();
    const transcriptionElem = document.getElementById("transcription");
    if (transcriptionElem) transcriptionElem.value = "";
    
    updateStatusMessage("Recording...", "green");
    logInfo("Recording started.");
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
// Clear any existing recording timer interval
if (recordingTimerInterval) {
  clearInterval(recordingTimerInterval);
  recordingTimerInterval = null;
}

recordingStartTime = Date.now();
recordingTimerInterval = setInterval(updateRecordingTimer, 1000);

      
      const track = mediaStream.getAudioTracks()[0];
      const processor = new MediaStreamTrackProcessor({ track: track });
      audioReader = processor.readable.getReader();
      
      function readLoop() {
        audioReader.read().then(({ done, value }) => {
          if (done) {
            logInfo("Audio track reading complete.");
            return;
          }
          lastFrameTime = Date.now();
          audioFrames.push(value);
          readLoop();
        }).catch(err => {
          logError("Error reading audio frames", err);
        });
      }
      readLoop();
      scheduleChunk();
      logInfo("MediaStreamTrackProcessor started, reading audio frames.");
      startButton.disabled = true;
      stopButton.disabled = false;
      pauseResumeButton.disabled = false;
      pauseResumeButton.innerText = "Pause Recording";
    } catch (error) {
      updateStatusMessage("Microphone access error: " + error, "red");
      logError("Microphone access error", error);
    }
  });

pauseResumeButton.addEventListener("click", async () => {
  // If there is no mediaStream or we're currently paused, then we want to RESUME recording.
  if (!mediaStream || recordingPaused) {
    try {
      // RESUME LOGIC: Request microphone access again
      mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const newTrack = mediaStream.getAudioTracks()[0];
      const processor = new MediaStreamTrackProcessor({ track: newTrack });
      audioReader = processor.readable.getReader();

      // Start the read loop for audio frames
      function readLoop() {
        audioReader.read().then(({ done, value }) => {
          if (done) {
            logInfo("Audio track reading complete.");
            return;
          }
          lastFrameTime = Date.now();
          audioFrames.push(value);
          readLoop();
        }).catch(err => {
          logError("Error reading audio frames", err);
        });
      }
      readLoop();

      // Reset timing variables for the resumed segment
      recordingPaused = false;
// Clear any existing recording timer interval
if (recordingTimerInterval) {
  clearInterval(recordingTimerInterval);
  recordingTimerInterval = null;
}

recordingStartTime = Date.now();
lastFrameTime = Date.now();
recordingTimerInterval = setInterval(updateRecordingTimer, 1000);

      
      // Update UI to reflect resumed state
      pauseResumeButton.innerText = "Pause Recording";
      updateStatusMessage("Recording...", "green");
      chunkStartTime = Date.now();
      scheduleChunk();
      logInfo("Recording resumed.");

      // Leave startButton and stopButton states unchanged; startButton remains disabled.
    } catch (error) {
      updateStatusMessage("Error resuming recording: " + error, "red");
      logError("Error resuming microphone on resume", error);
    }
  } else {
    // PAUSE LOGIC: Process pending audio, then stop the microphone so that the red dot goes away.
    await safeProcessAudioChunk(false);
    accumulatedRecordingTime += Date.now() - recordingStartTime;
    
    // Fully stop the media stream to release the microphone
    stopMicrophone();
    
    recordingPaused = true;
    clearInterval(recordingTimerInterval);
    clearTimeout(chunkTimeoutId);
    
    // Update UI to reflect paused state:
    pauseResumeButton.innerText = "Resume Recording";
    updateStatusMessage("Recording paused", "orange");
    logInfo("Recording paused; current chunk processed and media stream stopped.");

    // Ensure that the pause/resume button stays enabled while startButton remains disabled and stopButton remains enabled.
    const startButton = document.getElementById("startButton");
    const stopButton = document.getElementById("stopButton");
    if (startButton) startButton.disabled = true;         // Remain disabled
    if (stopButton) stopButton.disabled = false;            // Remain enabled
    pauseResumeButton.disabled = false;                    // Allow user to resume
  }
});

stopButton.addEventListener("click", async () => {
  updateStatusMessage("Finishing transcription...", "blue");
  manualStop = true;
  clearTimeout(chunkTimeoutId);
  clearInterval(recordingTimerInterval);
  stopMicrophone();
  chunkStartTime = 0;
  lastFrameTime = 0;
  await new Promise(resolve => setTimeout(resolve, 200));

  // NEW: If the recording is paused, finalize immediately.
  if (recordingPaused) {
    finalChunkProcessed = true;
    const compTimerElem = document.getElementById("transcribeTimer");
    if (compTimerElem) {
      compTimerElem.innerText = "Completion Timer: 0 sec";
    }
    updateStatusMessage("Transcription finished!", "green");
    const startButton = document.getElementById("startButton");
    if (startButton) startButton.disabled = false;
    stopButton.disabled = true;
    const pauseResumeButton = document.getElementById("pauseResumeButton");
    if (pauseResumeButton) pauseResumeButton.disabled = true;
    logInfo("Recording paused and stop pressed; transcription complete without extra processing.");
    return;
  }

  // Continue with the existing logic if not paused:
  if (audioFrames.length === 0 && !processedAnyAudioFrames) {
    resetRecordingState();
    if (completionTimerInterval) {
      clearInterval(completionTimerInterval);
      completionTimerInterval = null;
    }
    const compTimerElem = document.getElementById("transcribeTimer");
    if (compTimerElem) {
      compTimerElem.innerText = "Completion Timer: 0 sec";
    }
    const recTimerElem = document.getElementById("recordTimer");
    if (recTimerElem) {
      recTimerElem.innerText = "Recording Timer: 0 sec";
    }
    updateStatusMessage("Recording reset. Ready to start.", "green");
    const startButton = document.getElementById("startButton");
    if (startButton) startButton.disabled = false;
    stopButton.disabled = true;
    const pauseResumeButton = document.getElementById("pauseResumeButton");
    if (pauseResumeButton) pauseResumeButton.disabled = true;
    logInfo("No audio frames captured. Full reset performed.");
    processedAnyAudioFrames = false;
    return;
  } else {
    if (chunkProcessingLock) {
      pendingStop = true;
      logDebug("Chunk processing locked at stop; setting pendingStop.");
    } else {
      await safeProcessAudioChunk(true);
      if (!processedAnyAudioFrames) {
        resetRecordingState();
        if (completionTimerInterval) {
          clearInterval(completionTimerInterval);
          completionTimerInterval = null;
        }
        const compTimerElem = document.getElementById("transcribeTimer");
        if (compTimerElem) {
          compTimerElem.innerText = "Completion Timer: 0 sec";
        }
        const recTimerElem = document.getElementById("recordTimer");
        if (recTimerElem) {
          recTimerElem.innerText = "Recording Timer: 0 sec";
        }
        updateStatusMessage("Recording reset. Ready to start.", "green");
        const startButton = document.getElementById("startButton");
        if (startButton) startButton.disabled = false;
        stopButton.disabled = true;
        const pauseResumeButton = document.getElementById("pauseResumeButton");
        if (pauseResumeButton) pauseResumeButton.disabled = true;
        logInfo("No audio frames processed after safeProcessAudioChunk. Full reset performed.");
        processedAnyAudioFrames = false;
        return;
      } else {
        finalChunkProcessed = true;
        finalizeStop();
        logInfo("Stop button processed; final chunk handled.");
      }
    }
  }
});

}

export { initRecording };
