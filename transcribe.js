// api/transcribe.js
// Vercel serverless proxy → Mistral Voxtral Mini Transcribe
// Buffers the incoming multipart body (no streaming), then forwards.
// Works around Node Fetch's "duplex required" error.

const { Buffer } = require("buffer");

module.exports = async (req, res) => {
  // CORS (tighten origin later if you want)
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization, x-api-key");

  if (req.method === "OPTIONS") return res.status(204).end();
  if (req.method !== "POST") return res.status(405).json({ error: "Method not allowed" });

  // Accept user-supplied key via header
  const headerKey =
    req.headers["x-api-key"] ||
    (req.headers["authorization"] &&
      req.headers["authorization"].replace(/Bearer\s+/i, "").trim());

  if (!headerKey) {
    return res.status(400).json({ error: "Missing API key (send in x-api-key or Authorization: Bearer ...)" });
  }

  const contentType = req.headers["content-type"] || "";
  if (!contentType.toLowerCase().startsWith("multipart/form-data")) {
    return res.status(400).json({ error: "Content-Type must be multipart/form-data" });
  }

  try {
    // Buffer the entire incoming multipart body
    const chunks = [];
    for await (const chunk of req) chunks.push(chunk);
    const bodyBuffer = Buffer.concat(chunks);

    // Forward to Mistral
    const upstream = await fetch("https://api.mistral.ai/v1/audio/transcriptions", {
      method: "POST",
      headers: {
        "x-api-key": headerKey,
        "content-type": contentType,
        "accept": "application/json"
      },
      body: bodyBuffer, // ← buffered body (no duplex needed)
    });

    // Return JSON back to the browser
    const text = await upstream.text(); // keep as text to pass through errors verbatim
    res.status(upstream.status).send(text);
  } catch (e) {
    res.status(502).json({
      error: "Upstream request failed",
      detail: String(e?.message || e)
    });
  }
};

// IMPORTANT: keep body parser off so we can read the raw stream
module.exports.config = { api: { bodyParser: false } };
