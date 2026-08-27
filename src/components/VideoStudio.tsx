import React, { useState } from 'react';
import { RETRO_GAMES } from '../data/games';
import { RetroGame, VideoSettings } from '../types';
import { 
  Video, Smartphone, Monitor, Sparkles, Copy, Check, Sliders, 
  Layers, Play, Wand2, Trophy, Flame, Film, Zap, Eye
} from 'lucide-react';

export const VideoStudio: React.FC = () => {
  const [selectedGame, setSelectedGame] = useState<RetroGame>(RETRO_GAMES[0]);
  const [aspectRatio, setAspectRatio] = useState<'16:9' | '9:16'>('9:16');
  const [clipSeconds, setClipSeconds] = useState<number>(6);
  const [selectedIters, setSelectedIters] = useState<number[]>([1, 100, 500, 2000]);
  const [activePreviewIter, setActivePreviewIter] = useState<number>(500);
  const [crtFilter, setCrtFilter] = useState<boolean>(true);
  const [watermark, setWatermark] = useState<string>('@RetroAIArena');
  const [customHeadline, setCustomHeadline] = useState<string>('AI Learns to Fight: 10,000 Iterations');
  const [copiedScript, setCopiedScript] = useState<boolean>(false);
  const [copiedTitle, setCopiedTitle] = useState<boolean>(false);

  const makeReelCommand = `python make_progress_reel.py --game ${selectedGame.id} \\
  --checkpoint-dir ./checkpoints \\
  --iterations ${selectedIters.join(' ')} \\
  --clip-seconds ${clipSeconds} --clip-seconds-short ${Math.max(3, Math.floor(clipSeconds / 2))} \\
  --out-dir ./progress_reels`;

  const copyReelCommand = () => {
    navigator.clipboard.writeText(makeReelCommand.replace(/\\\s+/g, ' '));
    setCopiedScript(true);
    setTimeout(() => setCopiedScript(false), 2000);
  };

  const copyViralTitle = (t: string) => {
    navigator.clipboard.writeText(t);
    setCopiedTitle(true);
    setTimeout(() => setCopiedTitle(false), 2000);
  };

  const viralPackages = [
    {
      title: `I Trained an AI on ${selectedGame.name} for 10,000 Iterations... Can I Beat It?`,
      hook: "Day 1: It didn't know how to jump. Day 30: It reads my inputs in 1 frame and combos me into oblivion. Watch what happens when I challenge my own creation.",
      ctr: '14.8% CTR (Viral High)',
      tag: 'Challenge / Narrative'
    },
    {
      title: `AI vs. Human: The Ultimate ${selectedGame.name} Showdown`,
      hook: "I spent 3 weeks training a neural network from scratch with zero prior knowledge. Today, I sit down on Player 1. Who wins?",
      ctr: '12.4% CTR (High)',
      tag: '1v1 Esports Battle'
    },
    {
      title: `The Exact Moment My AI Discovered an Unstoppable Glitch in ${selectedGame.name}`,
      hook: "At Iteration 420, something bizarre happened in the neural weights. The AI found a frame trap that breaks the game entirely.",
      ctr: '16.2% CTR (Breakout)',
      tag: 'Discovery / Mystery'
    }
  ];

  return (
    <div className="space-y-6">
      {/* Top Header */}
      <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-xl backdrop-blur-sm">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div>
            <div className="flex items-center space-x-2">
              <Video className="w-5 h-5 text-rose-400" />
              <h2 className="text-lg font-bold text-white">Automated Video Reel & YouTube Studio</h2>
            </div>
            <p className="text-xs text-slate-400 mt-1">
              Assemble pixel-perfect progression reels, 9:16 Shorts/TikTok clips, CRT retro overlays, and high-retention viral packages.
            </p>
          </div>

          <div className="flex items-center space-x-2">
            <span className="text-xs text-slate-400 font-medium">Game:</span>
            <select
              value={selectedGame.id}
              onChange={e => {
                const g = RETRO_GAMES.find(x => x.id === e.target.value) || RETRO_GAMES[0];
                setSelectedGame(g);
              }}
              className="bg-slate-950 border border-slate-700 text-slate-200 text-xs rounded-xl px-3 py-2 focus:outline-none focus:border-rose-500 font-semibold"
            >
              {RETRO_GAMES.map(g => (
                <option key={g.id} value={g.id}>
                  {g.name} ({g.console})
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Main Studio Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left 6 Cols: Video Mockup Preview Canvas */}
        <div className="lg:col-span-6 space-y-4">
          <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-lg">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-bold uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
                <Film className="w-4 h-4 text-amber-400" />
                Live Video Rendering Preview
              </h3>

              {/* Format Switcher */}
              <div className="flex bg-slate-950 p-1 rounded-xl border border-slate-800">
                <button
                  onClick={() => setAspectRatio('9:16')}
                  className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                    aspectRatio === '9:16'
                      ? 'bg-rose-500 text-white shadow-sm'
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  <Smartphone className="w-3.5 h-3.5" />
                  <span>9:16 Shorts</span>
                </button>
                <button
                  onClick={() => setAspectRatio('16:9')}
                  className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                    aspectRatio === '16:9'
                      ? 'bg-rose-500 text-white shadow-sm'
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  <Monitor className="w-3.5 h-3.5" />
                  <span>16:9 YouTube</span>
                </button>
              </div>
            </div>

            {/* Video Canvas Container */}
            <div className="flex justify-center bg-black/70 p-4 rounded-xl border border-slate-800 overflow-hidden">
              <div
                className={`relative bg-slate-950 rounded-xl overflow-hidden border border-slate-700 shadow-2xl transition-all flex flex-col justify-between ${
                  aspectRatio === '9:16' ? 'w-[280px] h-[497px]' : 'w-full aspect-[16/9]'
                }`}
              >
                {/* Video Top Overlay */}
                <div className="p-3 bg-gradient-to-b from-black/90 to-transparent z-10">
                  <div className="bg-rose-600/90 backdrop-blur-md px-3 py-1.5 rounded-lg text-center shadow-lg border border-rose-400/40">
                    <span className="text-xs font-black tracking-wide text-white uppercase block truncate">
                      {customHeadline}
                    </span>
                  </div>
                </div>

                {/* Simulated Game Stage with Nearest Neighbor Sharp Pixels */}
                <div className="relative flex-1 flex items-center justify-center bg-indigo-950/40 overflow-hidden">
                  {/* Game Footage Simulation Box */}
                  <div className="w-full aspect-[4/3] bg-gradient-to-tr from-slate-950 via-slate-900 to-indigo-950 relative flex items-center justify-center p-4 border-y border-slate-800">
                    {/* Retro Fighter/Runner Silhouette */}
                    <div className="flex items-center justify-around w-full">
                      <div className="text-center">
                        <div className="w-12 h-16 bg-sky-400 rounded mx-auto mb-1 flex items-center justify-center font-black text-black text-xs">
                          P1
                        </div>
                        <span className="text-[9px] font-mono text-sky-300 font-bold">HUMAN</span>
                      </div>

                      <div className="text-center">
                        <div className="text-xs font-black text-amber-400 animate-pulse">VS</div>
                      </div>

                      <div className="text-center">
                        <div className="w-12 h-16 bg-rose-500 rounded mx-auto mb-1 flex items-center justify-center font-black text-white text-xs">
                          P2
                        </div>
                        <span className="text-[9px] font-mono text-rose-300 font-bold">AI (Iter {activePreviewIter})</span>
                      </div>
                    </div>

                    {/* Centered Iteration Stamp */}
                    <div className="absolute top-2 left-1/2 -translate-x-1/2 bg-black/80 px-3 py-1 rounded-md text-[11px] font-mono font-bold text-amber-400 border border-amber-500/40 tracking-wider">
                      ITERATION {activePreviewIter.toLocaleString()}
                    </div>

                    {/* CRT Scanline Overlay */}
                    {crtFilter && (
                      <div className="absolute inset-0 bg-[linear-gradient(rgba(18,16,16,0)_50%,rgba(0,0,0,0.25)_50%)] bg-[length:100%_4px] pointer-events-none" />
                    )}
                  </div>
                </div>

                {/* Video Bottom Overlay */}
                <div className="p-3 bg-gradient-to-t from-black/90 to-transparent z-10 flex items-center justify-between text-[11px] font-mono text-slate-300">
                  <span className="text-amber-400 font-bold">{watermark}</span>
                  <span className="bg-slate-900/90 px-2 py-0.5 rounded border border-slate-700">
                    {aspectRatio === '9:16' ? '1080x1920' : '1920x1080'} • 60 FPS
                  </span>
                </div>
              </div>
            </div>

            {/* Checkpoint Timeline Selector for preview */}
            <div className="mt-4">
              <span className="text-xs font-semibold text-slate-400 block mb-2">
                Preview Checkpoint Segment in Reel:
              </span>
              <div className="flex gap-2">
                {selectedIters.map(iter => (
                  <button
                    key={iter}
                    onClick={() => setActivePreviewIter(iter)}
                    className={`flex-1 py-1.5 px-2 rounded-lg text-xs font-mono font-bold transition-all border ${
                      activePreviewIter === iter
                        ? 'bg-rose-500 text-white border-rose-400 shadow-md shadow-rose-500/20'
                        : 'bg-slate-950 text-slate-400 border-slate-800 hover:border-slate-700'
                    }`}
                  >
                    Iter {iter}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Right 6 Cols: Customizer & Viral Package Generator */}
        <div className="lg:col-span-6 space-y-4">
          {/* Reel Settings */}
          <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-lg">
            <h3 className="text-sm font-bold uppercase tracking-wider text-slate-300 mb-4 flex items-center gap-1.5">
              <Sliders className="w-4 h-4 text-rose-400" />
              Reel Assembly Parameters
            </h3>

            <div className="space-y-4">
              <div>
                <label className="text-xs font-semibold text-slate-300 block mb-1">
                  Video Overlay Headline:
                </label>
                <input
                  type="text"
                  value={customHeadline}
                  onChange={e => setCustomHeadline(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-700 text-slate-200 text-xs rounded-xl px-3 py-2 focus:outline-none focus:border-rose-500 font-semibold"
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-xs font-semibold text-slate-300 block mb-1">
                    Seconds Per Checkpoint:
                  </label>
                  <input
                    type="number"
                    value={clipSeconds}
                    onChange={e => setClipSeconds(Math.max(2, parseInt(e.target.value) || 6))}
                    className="w-full bg-slate-950 border border-slate-700 text-slate-200 text-xs rounded-xl px-3 py-2 focus:outline-none focus:border-rose-500 font-mono"
                  />
                  <span className="text-[10px] text-slate-500 mt-1 block">
                    Total video: <strong>{clipSeconds * selectedIters.length}s</strong>
                  </span>
                </div>

                <div>
                  <label className="text-xs font-semibold text-slate-300 block mb-1">
                    Channel / Handle Watermark:
                  </label>
                  <input
                    type="text"
                    value={watermark}
                    onChange={e => setWatermark(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-700 text-slate-200 text-xs rounded-xl px-3 py-2 focus:outline-none focus:border-rose-500 font-mono"
                  />
                </div>
              </div>

              <div className="flex items-center justify-between pt-1">
                <label className="text-xs font-semibold text-slate-300 flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={crtFilter}
                    onChange={e => setCrtFilter(e.target.checked)}
                    className="rounded bg-slate-950 border-slate-700 text-rose-500 focus:ring-rose-500"
                  />
                  <span>Authentic CRT Scanline Shader Overlay</span>
                </label>
              </div>

              {/* CLI Command */}
              <div className="bg-slate-950 rounded-xl p-3.5 border border-slate-800 font-mono text-xs text-rose-300">
                <div className="flex items-center justify-between mb-1 text-[10px] text-slate-400 font-sans">
                  <span>Python ffmpeg assembly command:</span>
                  <button
                    onClick={copyReelCommand}
                    className="flex items-center gap-1 px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-white text-[10px] transition-all"
                  >
                    {copiedScript ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3 text-slate-400" />}
                    {copiedScript ? 'Copied' : 'Copy Command'}
                  </button>
                </div>
                <div className="overflow-x-auto leading-relaxed select-all">
                  {makeReelCommand}
                </div>
              </div>
            </div>
          </div>

          {/* High-CTR YouTube Title & Hook Packages */}
          <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-lg">
            <h3 className="text-sm font-bold uppercase tracking-wider text-slate-300 mb-3 flex items-center gap-1.5">
              <Wand2 className="w-4 h-4 text-amber-400" />
              High-CTR Titles & Short-Form Hooks
            </h3>

            <div className="space-y-3">
              {viralPackages.map((pkg, idx) => (
                <div key={idx} className="bg-slate-950 border border-slate-800 rounded-xl p-3 text-xs space-y-1.5">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] font-bold uppercase px-2 py-0.5 rounded bg-amber-500/20 text-amber-400 border border-amber-500/30 font-mono">
                      {pkg.ctr}
                    </span>
                    <button
                      onClick={() => copyViralTitle(pkg.title)}
                      className="text-[10px] text-slate-400 hover:text-white flex items-center gap-1"
                    >
                      <Copy className="w-3 h-3" /> Copy Title
                    </button>
                  </div>
                  <h4 className="font-bold text-slate-200 text-xs sm:text-sm">
                    {pkg.title}
                  </h4>
                  <p className="text-slate-400 text-[11px] leading-relaxed">
                    <strong>Hook:</strong> "{pkg.hook}"
                  </p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
