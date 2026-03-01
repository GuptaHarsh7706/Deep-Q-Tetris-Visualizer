import TetrisVisualizer from '@/components/TetrisVisualizer';

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-8 bg-gray-950 text-white">
      <div className="z-10 max-w-7xl w-full items-center justify-between font-mono text-sm">
        <h1 className="text-4xl font-bold mb-8 text-center text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-cyan-400">
          Tetris Deep Q-Network
        </h1>
        <p className="text-center text-gray-400 mb-8 max-w-2xl mx-auto">
          Input the number of episodes and watch the PyTorch RL model train in real-time. 
          The neural network pathways will illuminate based on the active node signals sent via WebSockets.
        </p>
        
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 shadow-2xl overflow-hidden">
          <TetrisVisualizer />
        </div>
      </div>
    </main>
  );
}
