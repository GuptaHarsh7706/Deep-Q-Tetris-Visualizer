"use client";

import React, { useEffect, useRef, useState } from 'react';

function connectionColor(weight: any, activation: any, wasCorrect: any) {
    if (Math.abs(activation) < 0.01 || Math.abs(weight) < 0.01) {
        return [15, 15, 15];
    }

    const magnitude = Math.tanh(Math.abs(weight) * Math.abs(activation) * 4.0);
    const brightness = Math.floor(magnitude * 255);

    if (brightness < 30) {
        return [15, 15, 15];
    }

    if (weight > 0 && wasCorrect) {
        return [0, Math.min(255, brightness + 80), Math.floor(brightness * 0.3)];
    } else if (weight > 0 && !wasCorrect) {
        return [Math.min(255, Math.floor(brightness * 0.7) + 50), Math.min(255, brightness + 50), 0];
    } else if (weight < 0 && wasCorrect) {
        return [Math.min(255, brightness + 60), Math.floor(brightness * 0.5), 0];
    } else {
        return [Math.min(255, brightness + 80), 0, Math.floor(brightness * 0.15)];
    }
}

export default function TetrisVisualizer() {
    const gameCanvasRef = useRef<any>(null);
    const netCanvasRef = useRef<any>(null);
    const wsRef = useRef<any>(null);

    const [episodes, setEpisodes] = useState<any>(10);
    const [status, setStatus] = useState("Disconnected");
    const [gameState, setGameState] = useState<any>(null);

    useEffect(() => {
        connectWebSocket();
        return () => {
            if (wsRef.current) wsRef.current.close();
        };
    }, []);

    const connectWebSocket = () => {
        setStatus("Connecting...");
        const wsUrl = process.env.NEXT_PUBLIC_WS_URL || "wss://deep-q-tetris-visualizer.onrender.com/ws/train";
        const ws = new WebSocket(wsUrl);

        ws.onopen = () => setStatus("Connected - Ready to Train");
        ws.onclose = () => setStatus("Disconnected");
        ws.onerror = (error) => {
            console.error("WebSocket Error:", error);
            setStatus("Error Connecting");
        };

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.status === "complete") {
                setStatus("Training Complete");
                return;
            }

            if (data.type === "frame") {
                setStatus(`Training... Episode ${data.game.episode}/${data.game.total_episodes}`);
                setGameState(data.game);
                drawGame(data.game);
                drawNetwork(data.network);
            }
        };

        wsRef.current = ws;
    };

    const startTraining = () => {
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({ episodes: Number(episodes) }));
            setStatus("Starting...");
        } else {
            connectWebSocket();
            setTimeout(() => startTraining(), 1000);
        }
    };

    const drawGame = (game: any) => {
        const ctx = gameCanvasRef.current.getContext('2d');
        const cols = 10;
        const rows = 20;
        const cellSize = 30;
        const colors = [
            '#00FFFF', // I - Cyan
            '#FFFF00', // O - Yellow
            '#800080', // T - Purple
            '#00FF00', // S - Green
            '#FF0000', // Z - Red
            '#0000FF', // J - Blue
            '#FFA500'  // L - Orange
        ];

        ctx.fillStyle = '#141414';
        ctx.fillRect(0, 0, cols * cellSize, rows * cellSize);

        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
                const val = game.board[r][c];
                const x = c * cellSize;
                const y = r * cellSize;

                if (val > 0) {
                    ctx.fillStyle = colors[val - 1];
                    ctx.fillRect(x, y, cellSize - 1, cellSize - 1);
                } else {
                    ctx.fillStyle = '#282828';
                    ctx.fillRect(x, y, cellSize - 1, cellSize - 1);
                }
                ctx.strokeStyle = '#3c3c3c';
                ctx.strokeRect(x, y, cellSize, cellSize);
            }
        }

        const TETROMINOES: any = {
            'I': [
                [[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]],
                [[0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]],
            ],
            'O': [
                [[1, 1], [1, 1]],
                [[1, 1], [1, 1]],
                [[1, 1], [1, 1]],
                [[1, 1], [1, 1]],
            ],
            'T': [
                [[0, 1, 0], [1, 1, 1], [0, 0, 0]],
                [[0, 1, 0], [0, 1, 1], [0, 1, 0]],
                [[0, 0, 0], [1, 1, 1], [0, 1, 0]],
                [[0, 1, 0], [1, 1, 0], [0, 1, 0]],
            ],
            'S': [
                [[0, 1, 1], [1, 1, 0], [0, 0, 0]],
                [[0, 1, 0], [0, 1, 1], [0, 0, 1]],
                [[0, 0, 0], [0, 1, 1], [1, 1, 0]],
                [[1, 0, 0], [1, 1, 0], [0, 1, 0]],
            ],
            'Z': [
                [[1, 1, 0], [0, 1, 1], [0, 0, 0]],
                [[0, 0, 1], [0, 1, 1], [0, 1, 0]],
                [[0, 0, 0], [1, 1, 0], [0, 1, 1]],
                [[0, 1, 0], [1, 1, 0], [1, 0, 0]],
            ],
            'J': [
                [[1, 0, 0], [1, 1, 1], [0, 0, 0]],
                [[0, 1, 1], [0, 1, 0], [0, 1, 0]],
                [[0, 0, 0], [1, 1, 1], [0, 0, 1]],
                [[0, 1, 0], [0, 1, 0], [1, 1, 0]],
            ],
            'L': [
                [[0, 0, 1], [1, 1, 1], [0, 0, 0]],
                [[0, 1, 0], [0, 1, 0], [0, 1, 1]],
                [[0, 0, 0], [1, 1, 1], [1, 0, 0]],
                [[1, 1, 0], [0, 1, 0], [0, 1, 0]],
            ]
        };

        const PIECE_NAMES = ['I', 'O', 'T', 'S', 'Z', 'J', 'L'];

        if (game.current_piece) {
            const shape = TETROMINOES[game.current_piece][game.current_rotation];
            const colorIdx = PIECE_NAMES.indexOf(game.current_piece);

            for (let r = 0; r < shape.length; r++) {
                for (let c = 0; c < shape[r].length; c++) {
                    if (shape[r][c]) {
                        const boardR = game.current_piece_row + r;
                        const boardC = game.current_piece_col + c;

                        if (boardR >= 0 && boardR < rows && boardC >= 0 && boardC < cols) {
                            const x = boardC * cellSize;
                            const y = boardR * cellSize;

                            ctx.fillStyle = colors[colorIdx];
                            ctx.fillRect(x, y, cellSize - 1, cellSize - 1);
                            ctx.strokeStyle = '#FFFFFF'; // White border for active piece
                            ctx.strokeRect(x, y, cellSize, cellSize);
                        }
                    }
                }
            }
        }
    };

    const drawNetwork = (net: any) => {
        const ctx = netCanvasRef.current.getContext('2d');
        const W = 800;
        const H = 800;

        ctx.fillStyle = '#000000';
        ctx.fillRect(0, 0, W, H);

        const layerSizes = [27, 128, 128, 5];
        const displaySizes = [27, 32, 32, 5];
        const padX = 80;
        const padY = 30;

        const layerXs = layerSizes.map((_, i) => padX + i * (W - 2 * padX) / 3);
        const neuronPos = displaySizes.map((size, l) => {
            const spacing = (H - 2 * padY) / Math.max(size, 1);
            const startY = padY + (H - 2 * padY - spacing * (size - 1)) / 2;
            return Array.from({ length: size }).map((_, i) => ({
                x: layerXs[l],
                y: startY + i * spacing
            }));
        });

        const { weights, activations, action, was_correct } = net;
        ctx.lineWidth = 1.0;

        for (let l = 0; l < 3; l++) {
            const w = weights[l];     // shape: [out, in]
            const acts = activations[l];
            const stride = (l === 1) ? 4 : 1;
            const srcPos = neuronPos[l];
            const dstPos = neuronPos[l + 1];

            for (let out_i = 0; out_i < w.length; out_i += stride) {
                for (let in_i = 0; in_i < w[0].length; in_i += stride) {
                    const weight = w[out_i][in_i];
                    const act = acts[in_i] || 0.0;

                    if (Math.abs(act) < 0.01) continue;

                    const mag = Math.tanh(Math.abs(weight) * Math.abs(act) * 3.0);
                    if (mag < 0.1) continue;

                    const src_d = w[0].length > 32 ? Math.floor(in_i / 4) : in_i;
                    const dst_d = w.length > 32 ? Math.floor(out_i / 4) : out_i;

                    if (src_d >= srcPos.length || dst_d >= dstPos.length) continue;

                    const rgb = connectionColor(weight, act, was_correct);
                    ctx.strokeStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},0.8)`;

                    const src = srcPos[src_d];
                    const dst = dstPos[dst_d];

                    ctx.beginPath();
                    ctx.moveTo(src.x, src.y);
                    ctx.lineTo(dst.x, dst.y);
                    ctx.stroke();

                    if (mag > 0.5) {
                        ctx.strokeStyle = `rgba(${Math.min(255, rgb[0] + 80)},${Math.min(255, rgb[1] + 80)},${Math.min(255, rgb[2] + 80)},0.8)`;
                        ctx.stroke();
                    }
                }
            }
        }

        for (let l = 0; l < 4; l++) {
            const acts = activations[l];
            const positions = neuronPos[l];
            const stride = displaySizes[l] < layerSizes[l] ? 4 : 1;

            positions.forEach((pos, i) => {
                let act = 0;
                if (stride === 1) {
                    act = acts[i] || 0;
                } else {
                    const slice = acts.slice(i * 4, i * 4 + 4);
                    act = slice.reduce((a: any, b: any) => a + b, 0) / slice.length;
                }

                let brightness = Math.min(255, Math.max(100, Math.floor(Math.abs(act) * 200) + 80));

                if (l === 3 && i === action) {
                    ctx.fillStyle = was_correct ? '#00FF50' : '#FF2828';
                    ctx.fillRect(pos.x - 7, pos.y - 7, 14, 14);
                    ctx.strokeStyle = '#FFFFFF';
                    ctx.strokeRect(pos.x - 9, pos.y - 9, 18, 18);
                } else {
                    ctx.fillStyle = `rgb(${brightness}, ${brightness}, ${brightness})`;
                    ctx.fillRect(pos.x - 4, pos.y - 4, 8, 8);
                }
            });
        }
    };

    return (
        <div className="flex flex-col gap-6">
            { }
            <div className="flex flex-wrap items-center gap-4 bg-gray-800 p-4 rounded-lg">
                <label className="font-bold">Episodes:</label>
                <input
                    type="number"
                    value={episodes}
                    onChange={(e) => setEpisodes(e.target.value)}
                    className="bg-gray-700 text-white rounded px-3 py-1 w-24 outline-none"
                />
                <button
                    onClick={startTraining}
                    className="bg-emerald-600 hover:bg-emerald-500 text-white font-bold py-1 px-4 rounded transition-colors"
                >
                    Start Training
                </button>
                <span className={`w-full sm:w-auto sm:ml-auto font-mono text-sm break-all ${status.includes("Connected") || status.includes("Training") ? "text-emerald-400" : "text-amber-400"}`}>
                    Status: {status}
                </span>
            </div>

            { }
            <div className="flex flex-col xl:flex-row gap-4 xl:h-[800px]">
                { }
                <div className="flex flex-col gap-2 w-full xl:w-[300px] max-w-[300px] mx-auto xl:mx-0">
                    <canvas
                        ref={gameCanvasRef}
                        width={300}
                        height={600}
                        className="rounded shadow-lg w-full h-auto bg-[#141414]"
                    />
                    {gameState && (
                        <div className="bg-gray-800 p-4 rounded flex flex-col gap-1 text-sm font-mono mt-auto xl:h-[192px]">
                            <div>Score: {Math.floor(gameState.score)}</div>
                            <div>Lines: {gameState.lines}</div>
                            <div>Epsilon: {gameState.epsilon.toFixed(4)}</div>
                            <div>Steps: {gameState.steps}</div>
                            <div>Buffer: {gameState.buffer_size} / 20000</div>
                        </div>
                    )}
                </div>

                { }
                <div className="flex-1 w-full max-w-[800px] bg-black rounded shadow-lg overflow-hidden relative border border-gray-800 mx-auto xl:mx-0 aspect-square xl:aspect-auto xl:h-[800px]">
                    <div className="absolute top-2 w-full flex flex-row justify-between px-4 sm:px-16 text-gray-400 font-mono text-xs z-10">
                        <span>Input</span>
                        <span>Hidden 1</span>
                        <span>Hidden 2</span>
                        <span>Output</span>
                    </div>
                    <canvas ref={netCanvasRef} width={800} height={800} className="w-full h-full object-cover sm:object-contain" />
                </div>
            </div>
        </div>
    );
}
