"use client";
import { useEffect, useRef, useState } from "react";

type PredictResponse = {
  prediction: number;
  confidence: number;
  probs: number[];
};

export default function DigitCanvas() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const ctxRef = useRef<CanvasRenderingContext2D | null>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const API_URL = "https://ai-digit-recognizer.onrender.com"; // TEMP for testing

  if (!API_URL) {
    console.warn("NEXT_PUBLIC_API_URL (or _API_BASE) is missing!");
  } else {
    console.log("Using API_URL:", API_URL);
  }

  // ----- setup the canvas once
  useEffect(() => {
    const c = canvasRef.current!;
    c.width = 280;
    c.height = 280;
    const ctx = c.getContext("2d")!;
    // white background, black pen
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, c.width, c.height);
    ctx.lineCap = "round";
    ctx.strokeStyle = "black";
    ctx.lineWidth = 20;
    ctxRef.current = ctx;
  }, []);

  // ----- helpers
  const mousePos = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const r = canvasRef.current!.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  };
  const touchPos = (e: React.TouchEvent<HTMLCanvasElement>) => {
    const t = e.touches[0];
    const r = canvasRef.current!.getBoundingClientRect();
    return { x: t.clientX - r.left, y: t.clientY - r.top };
  };

  // ----- mouse drawing
  const onMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    setIsDrawing(true);
    const { x, y } = mousePos(e);
    ctxRef.current!.beginPath();
    ctxRef.current!.moveTo(x, y);
  };
  const onMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing) return;
    const { x, y } = mousePos(e);
    ctxRef.current!.lineTo(x, y);
    ctxRef.current!.stroke();
  };
  const onMouseUp = () => {
    setIsDrawing(false);
    ctxRef.current!.closePath();
  };

  // ----- touch drawing (phones/tablets)
  const onTouchStart = (e: React.TouchEvent<HTMLCanvasElement>) => {
    e.preventDefault(); // don't scroll page while drawing
    setIsDrawing(true);
    const { x, y } = touchPos(e);
    ctxRef.current!.beginPath();
    ctxRef.current!.moveTo(x, y);
  };
  const onTouchMove = (e: React.TouchEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    if (!isDrawing) return;
    const { x, y } = touchPos(e);
    ctxRef.current!.lineTo(x, y);
    ctxRef.current!.stroke();
  };
  const onTouchEnd = () => {
    setIsDrawing(false);
    ctxRef.current!.closePath();
  };

  // ----- actions
  const clearCanvas = () => {
    const c = canvasRef.current!;
    ctxRef.current!.fillStyle = "white";
    ctxRef.current!.fillRect(0, 0, c.width, c.height);
    setResult(null);
  };

  const predict = async () => {
    setResult(null);

    // downscale to 28x28
    const big = canvasRef.current!;
    const tmp = document.createElement("canvas");
    tmp.width = 28;
    tmp.height = 28;
    const tctx = tmp.getContext("2d")!;
    tctx.drawImage(big, 0, 0, 28, 28);

    // send as file (no headers)
    tmp.toBlob(async (blob) => {
      if (!blob) {
        alert("Could not read canvas");
        return;
      }
      const form = new FormData();
      form.append("file", blob, "digit.png");

      try {
        const res = await fetch(`${API_URL}/predict`, {
          method: "POST",
          body: form, // ← no Content-Type header
        });
        const text = await res.text(); // handy for debugging
        if (!res.ok) throw new Error(text || `HTTP ${res.status}`);
        const json = JSON.parse(text);
        setResult(json);
      } catch (e: any) {
        console.error("Predict failed:", e);
        alert(`Predict failed: ${e.message || e}`);
      }
    }, "image/png");
  };

  return (
    <div className="max-w-xl mx-auto space-y-4">
      <h1 className="text-2xl font-semibold">AI Digit Recognizer</h1>
      <p className="text-sm text-gray-600">
        Draw a digit (0–9), then click Predict.
      </p>

      <canvas
        ref={canvasRef}
        className="border rounded shadow"
        style={{ touchAction: "none" }} // <— stops touch scrolling
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={onMouseUp}
        onTouchStart={onTouchStart}
        onTouchMove={onTouchMove}
        onTouchEnd={onTouchEnd}
      />

      <div className="flex gap-2">
        <button
          onClick={predict}
          className="px-4 py-2 rounded bg-black text-white"
        >
          Predict
        </button>
        <button onClick={clearCanvas} className="px-4 py-2 rounded border">
          Clear
        </button>
      </div>

      {/* --- Confidence bar chart (0–9) --- */}
      {result && Array.isArray(result.probs) && (
        <div className="p-3 border rounded space-y-2">
          <div className="font-medium">
            Prediction:{" "}
            <span className="text-blue-600">{result.prediction}</span>
            <span className="ml-2 text-sm text-gray-600">
              ({(result.confidence * 100).toFixed(1)}%)
            </span>
          </div>

          <div className="space-y-1">
            {result.probs.map((p, i) => (
              <div key={i} className="flex items-center gap-2">
                <div className="w-6 text-right text-xs">{i}</div>
                <div className="flex-1 h-2 bg-gray-200 rounded">
                  <div
                    className="h-2 rounded bg-blue-500"
                    style={{ width: `${(p * 100).toFixed(1)}%` }}
                  />
                </div>
                <div className="w-12 text-xs text-right">
                  {(p * 100).toFixed(1)}%
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
