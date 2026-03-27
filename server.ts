import express from "express";
import path from "path";
import { createServer as createViteServer } from "vite";
import multer from "multer";
import cors from "cors";
import { spawn } from "child_process";
import fs from "fs";

const app = express();
const PORT = Number(process.env.PORT) || 3000;

// Configure multer for image uploads
const storage = multer.diskStorage({
  destination: (req: any, file: any, cb: any) => {
    const uploadDir = process.env.NODE_ENV === "production" 
      ? "/tmp" 
      : path.join(process.cwd(), "uploads");
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }
    cb(null, uploadDir);
  },
  filename: (req: any, file: any, cb: any) => {
    cb(null, `${Date.now()}-${file.originalname}`);
  },
});
const upload = multer({ storage });

app.use(cors());
app.use(express.json());

// API routes
app.post("/api/predict", upload.single("image"), (req: any, res: any) => {
  if (!req.file) {
    return res.status(400).json({ error: "No image file uploaded" });
  }

  const { age, gender, smoking_years } = req.body;
  const imagePath = req.file.path;

  // Call Python script for inference
  // We'll pass the image path and metadata as arguments
  // On Windows, the command is usually 'python', on Linux/Mac 'python3'
  let pythonCmd = process.platform === "win32" ? "python" : "python3";
  
  // Robust check: if python3 fails, try python
  try {
    const { execSync } = require("child_process");
    if (process.platform !== "win32") {
      try {
        execSync("python3 --version", { stdio: "ignore" });
      } catch (e) {
        pythonCmd = "python";
      }
    }
  } catch (e) {
    // Fallback to default
  }
  
  console.log(`Using Python command: ${pythonCmd}`);

  const pythonProcess = spawn(pythonCmd, [
    path.join(process.cwd(), "backend", "predict.py"),
    imagePath,
    age || "0",
    gender || "0",
    smoking_years || "0",
  ]);

  let result = "";
  let error = "";

  pythonProcess.stdout.on("data", (data) => {
    result += data.toString();
  });

  pythonProcess.stderr.on("data", (data) => {
    const msg = data.toString();
    error += msg;
    process.stderr.write(msg); // Log to main terminal in real-time
  });

  pythonProcess.on("close", (code) => {
    // Clean up uploaded file
    try {
      if (fs.existsSync(imagePath)) {
        fs.unlinkSync(imagePath);
      }
    } catch (e) {
      console.error("Error deleting file:", e);
    }

    if (code !== 0) {
      console.error("Python script error:", error);
      return res.status(500).json({ error: "Prediction failed", details: error });
    }

    try {
      const prediction = JSON.parse(result);
      res.json(prediction);
    } catch (e) {
      console.error("Error parsing Python output:", e);
      res.status(500).json({ error: "Invalid prediction result", details: result });
    }
  });
});

async function startServer() {
  // Vite middleware for development
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
