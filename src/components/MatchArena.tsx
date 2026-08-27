import React, { useState, useEffect, useRef } from 'react';
import { RetroGame, GameMode, MatchState } from '../types';
import { RETRO_GAMES } from '../data/games';
import confetti from 'canvas-confetti';
import { 
  Play, Pause, RotateCcw, Copy, Check, Swords, Shield, Zap, Sparkles, 
  Gamepad2, Cpu, User, Flame, Eye, Video, Trophy, ArrowRight, Activity, Terminal
} from 'lucide-react';

export const MatchArena: React.FC = () => {
  const [selectedGame, setSelectedGame] = useState<RetroGame>(RETRO_GAMES[0]);
  const [mode, setMode] = useState<GameMode>('versus');
  const [selectedPresetIdx, setSelectedPresetIdx] = useState<number>(2); // default iter 500
  const [copiedCmd, setCopiedCmd] = useState<boolean>(false);
  const [activeKeys, setActiveKeys] = useState<Record<string, boolean>>({});

  // Match State
  const [match, setMatch] = useState<MatchState>({
    gameId: selectedGame.id,
    mode: 'versus',
    stateName: selectedGame.defaultState,
    p1Name: 'HUMAN (You)',
    p1Health: 100,
    p2Name: selectedGame.aiDifficultyPresets[2].title,
    p2Health: 100,
    p1Score: 12400,
    p2Score: 8900,
    p1Wins: 0,
    p2Wins: 0,
    currentRound: 1,
    timer: 99,
    isPlaying: false,
    isPaused: false,
    matchOver: false,
    winner: null,
    p1LastAction: 'IDLE',
    p2LastAction: 'GUARD',
    aiProbabilities: [
      { action: 'LOW_KICK', prob: 42 },
      { action: 'UPPERCUT', prob: 28 },
      { action: 'PROJECTILE', prob: 18 },
      { action: 'BLOCK', prob: 12 }
    ],
    p1CombosCount: 0,
    p2CombosCount: 0,
    stepCount: 0
  });

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const preset = selectedGame.aiDifficultyPresets[selectedPresetIdx] || selectedGame.aiDifficultyPresets[0];

  // Sync game changes
  const handleSelectGame = (game: RetroGame) => {
    setSelectedGame(game);
    const validModes = game.supportedModes;
    const newMode = validModes.includes(mode) ? mode : validModes[0];
    setMode(newMode);
    setSelectedPresetIdx(Math.min(selectedPresetIdx, game.aiDifficultyPresets.length - 1));
    resetMatch(game, newMode, 0);
  };

  const resetMatch = (game = selectedGame, currentMode = mode, presetIdx = selectedPresetIdx) => {
    const curPreset = game.aiDifficultyPresets[presetIdx] || game.aiDifficultyPresets[0];
    setMatch({
      gameId: game.id,
      mode: currentMode,
      stateName: game.defaultState,
      p1Name: 'HUMAN (Player 1)',
      p1Health: 100,
      p2Name: curPreset.title,
      p2Health: 100,
      p1Score: 0,
      p2Score: 0,
      p1Wins: 0,
      p2Wins: 0,
      currentRound: 1,
      timer: 99,
      isPlaying: false,
      isPaused: false,
      matchOver: false,
      winner: null,
      p1LastAction: 'READY',
      p2LastAction: 'READY',
      aiProbabilities: [
        { action: 'APPROACH', prob: 50 },
        { action: 'ATTACK', prob: 30 },
        { action: 'BLOCK', prob: 20 }
      ],
      p1CombosCount: 0,
      p2CombosCount: 0,
      stepCount: 0
    });
  };

  // Keyboard capture for Human Player 1
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const k = e.key.toLowerCase();
      setActiveKeys(prev => ({ ...prev, [k]: true }));

      if (!match.isPlaying || match.matchOver) return;

      let humanAction = '';
      let damage = 0;

      if (k === 'z' || k === 'j') {
        humanAction = 'PUNCH';
        damage = Math.floor(Math.random() * 8) + 4;
      } else if (k === 'x' || k === 'k') {
        humanAction = 'KICK / JUMP';
        damage = Math.floor(Math.random() * 12) + 6;
      } else if (k === 'c' || k === 'l') {
        humanAction = 'SPECIAL MOVE!';
        damage = Math.floor(Math.random() * 18) + 12;
      } else if (k === 'a' || k === 's') {
        humanAction = 'BLOCK';
      } else if (k === 'arrowright' || k === 'd') {
        humanAction = 'FORWARD SPRINT';
      } else if (k === 'arrowleft' || k === 'a') {
        humanAction = 'BACKSTEP';
      } else if (k === 'arrowup' || k === 'w') {
        humanAction = 'AIR HOP';
      } else if (k === 'arrowdown' || k === 's') {
        humanAction = 'CROUCH';
      }

      if (humanAction) {
        // Execute Human action
        setMatch(prev => {
          if (prev.matchOver) return prev;
          const newP2Health = Math.max(0, prev.p2Health - (humanAction === 'BLOCK' ? 0 : damage));
          const newP1Score = prev.p1Score + (damage * 100);
          const isP2Dead = newP2Health <= 0;

          if (isP2Dead) {
            confetti({ particleCount: 100, spread: 70, origin: { y: 0.6 } });
          }

          return {
            ...prev,
            p1LastAction: humanAction,
            p2Health: newP2Health,
            p1Score: newP1Score,
            p1CombosCount: humanAction === 'SPECIAL MOVE!' ? prev.p1CombosCount + 1 : prev.p1CombosCount,
            matchOver: isP2Dead,
            winner: isP2Dead ? 'p1' : prev.winner,
            isPlaying: isP2Dead ? false : prev.isPlaying
          };
        });
      }
    };

    const handleKeyUp = (e: KeyboardEvent) => {
      const k = e.key.toLowerCase();
      setActiveKeys(prev => ({ ...prev, [k]: false }));
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, [match.isPlaying, match.matchOver]);

  // AI Game Loop simulation
  useEffect(() => {
    if (!match.isPlaying || match.matchOver || match.isPaused) return;

    const interval = setInterval(() => {
      setMatch(prev => {
        if (prev.matchOver || !prev.isPlaying) return prev;

        // Decrease timer
        const newTimer = prev.timer > 0 ? prev.timer - 1 : 0;
        if (newTimer === 0) {
          const winner = prev.p1Health > prev.p2Health ? 'p1' : (prev.p2Health > prev.p1Health ? 'p2' : 'tie');
          if (winner === 'p1') confetti({ particleCount: 120, spread: 80 });
          return {
            ...prev,
            timer: 0,
            isPlaying: false,
            matchOver: true,
            winner
          };
        }

        // AI reaction calculation based on preset skill
        const aiSkill = preset.winRateVsHuman / 100; // 0.03 to 0.95
        const willAiAttack = Math.random() < (0.3 + (aiSkill * 0.5));
        let aiAction = 'OBSERVE';
        let p1DamageTaken = 0;

        if (willAiAttack) {
          const attacks = ['FRAME_TRAP_PUNCH', 'SWEEP_KICK', 'HADOUKEN_BLAST', 'AIR_GRAB', 'UPPERCUT'];
          aiAction = attacks[Math.floor(Math.random() * attacks.length)];
          const baseDamage = Math.floor(Math.random() * 10) + 4;
          p1DamageTaken = Math.floor(baseDamage * (0.8 + aiSkill * 0.8));
        } else {
          aiAction = Math.random() > 0.5 ? 'GUARD_STANCE' : 'MICRO_SPACING';
        }

        // Generate synthetic CNN softmax distribution
        const pAtk = Math.min(95, Math.floor(willAiAttack ? 55 + aiSkill * 35 : 15));
        const pBlock = Math.floor((100 - pAtk) * 0.6);
        const pMove = 100 - pAtk - pBlock;

        const newP1Health = Math.max(0, prev.p1Health - p1DamageTaken);
        const isP1Dead = newP1Health <= 0;

        return {
          ...prev,
          timer: newTimer,
          p1Health: newP1Health,
          p2Score: prev.p2Score + (p1DamageTaken * 80),
          p2LastAction: aiAction,
          p2CombosCount: willAiAttack ? prev.p2CombosCount + 1 : prev.p2CombosCount,
          stepCount: prev.stepCount + 1,
          aiProbabilities: [
            { action: aiAction, prob: pAtk },
            { action: 'COUNTER_PARRY', prob: pBlock },
            { action: 'PIVOT_SPACING', prob: pMove }
          ],
          matchOver: isP1Dead,
          winner: isP1Dead ? 'p2' : prev.winner,
          isPlaying: isP1Dead ? false : prev.isPlaying
        };
      });
    }, 450);

    return () => clearInterval(interval);
  }, [match.isPlaying, match.matchOver, match.isPaused, preset]);

  // Canvas visualizer render
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animId: number;
    let frame = 0;

    const render = () => {
      frame++;
      const w = canvas.width;
      const h = canvas.height;

      // Background gradient
      const bgGrad = ctx.createLinearGradient(0, 0, 0, h);
      bgGrad.addColorStop(0, '#0f172a');
      bgGrad.addColorStop(0.65, '#1e1b4b');
      bgGrad.addColorStop(1, '#020617');
      ctx.fillStyle = bgGrad;
      ctx.fillRect(0, 0, w, h);

      // Floor grid / Arena stage
      ctx.strokeStyle = '#312e81';
      ctx.lineWidth = 1;
      const floorY = h - 60;
      ctx.beginPath();
      ctx.moveTo(0, floorY);
      ctx.lineTo(w, floorY);
      ctx.stroke();

      for (let x = 0; x < w; x += 40) {
        ctx.beginPath();
        ctx.moveTo(x, floorY);
        ctx.lineTo(x - 20 + ((frame * 0.5) % 40), h);
        ctx.stroke();
      }

      // Fighter 1 (Human - Blue/Cyan)
      const p1X = 120 + (activeKeys['d'] || activeKeys['arrowright'] ? 30 : 0) - (activeKeys['a'] || activeKeys['arrowleft'] ? 20 : 0);
      const p1Y = floorY - 60 - (activeKeys['w'] || activeKeys['arrowup'] ? 40 : 0) + (activeKeys['s'] || activeKeys['arrowdown'] ? 20 : 0);
      
      // Shadow
      ctx.fillStyle = 'rgba(0,0,0,0.5)';
      ctx.beginPath();
      ctx.ellipse(p1X + 20, floorY + 5, 25, 8, 0, 0, Math.PI * 2);
      ctx.fill();

      // P1 Body
      ctx.fillStyle = '#38bdf8';
      ctx.fillRect(p1X, p1Y, 32, 54);
      // P1 Head
      ctx.fillStyle = '#fed7aa';
      ctx.fillRect(p1X + 6, p1Y - 20, 20, 20);
      // P1 Headband/Cap
      ctx.fillStyle = '#ef4444';
      ctx.fillRect(p1X + 4, p1Y - 20, 24, 6);

      // Fighter 2 (AI - Crimson/Purple)
      const p2X = w - 160;
      const p2Y = floorY - 60;

      // P2 Shadow
      ctx.fillStyle = 'rgba(0,0,0,0.5)';
      ctx.beginPath();
      ctx.ellipse(p2X + 20, floorY + 5, 25, 8, 0, 0, Math.PI * 2);
      ctx.fill();

      // P2 Body
      ctx.fillStyle = '#f43f5e';
      ctx.fillRect(p2X, p2Y, 32, 54);
      // P2 Head
      ctx.fillStyle = '#fed7aa';
      ctx.fillRect(p2X + 6, p2Y - 20, 20, 20);
      // P2 Mask / Visor (AI eye glow)
      ctx.fillStyle = '#10b981';
      ctx.fillRect(p2X + 4, p2Y - 14, 16, 5);

      // Hit sparks / attack waves
      if (match.isPlaying) {
        if (match.p1LastAction.includes('PUNCH') || match.p1LastAction.includes('SPECIAL')) {
          ctx.fillStyle = '#38bdf8';
          ctx.beginPath();
          ctx.arc(p1X + 50, p1Y + 20, 16 + Math.sin(frame * 0.5) * 8, 0, Math.PI * 2);
          ctx.fill();
        }
        if (match.p2LastAction.includes('PUNCH') || match.p2LastAction.includes('HADOUKEN')) {
          ctx.fillStyle = '#f43f5e';
          ctx.beginPath();
          ctx.arc(p2X - 25, p2Y + 20, 16 + Math.cos(frame * 0.5) * 8, 0, Math.PI * 2);
          ctx.fill();
        }
      }

      // CRT Scanline Shader effect
      ctx.fillStyle = 'rgba(0, 0, 0, 0.15)';
      for (let y = 0; y < h; y += 3) {
        ctx.fillRect(0, y, w, 1);
      }

      animId = requestAnimationFrame(render);
    };

    render();
    return () => cancelAnimationFrame(animId);
  }, [match.isPlaying, match.p1LastAction, match.p2LastAction, activeKeys]);

  // CLI Command Generator
  const generatedCommand = `python play_human_vs_ai.py --game ${selectedGame.id} \\
    --model ./checkpoints/${selectedGame.id}/iter_${preset.iter}.zip \\
    --mode ${mode} --scale 4 --fps 60`;

  const copyCommand = () => {
    navigator.clipboard.writeText(generatedCommand.replace(/\\\s+/g, ' '));
    setCopiedCmd(true);
    setTimeout(() => setCopiedCmd(false), 2000);
  };

  return (
    <div className="space-y-6">
      {/* Top Banner / Matchup Selector */}
      <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-xl backdrop-blur-sm">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
          {/* Game Selection */}
          <div className="flex-1">
            <label className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-2 block flex items-center gap-1.5">
              <Gamepad2 className="w-4 h-4 text-rose-400" />
              Select Retro Title & Genre
            </label>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-2">
              {RETRO_GAMES.map(g => (
                <button
                  key={g.id}
                  onClick={() => handleSelectGame(g)}
                  className={`px-3 py-2.5 rounded-xl text-left transition-all border ${
                    selectedGame.id === g.id
                      ? 'bg-rose-500/20 border-rose-500 text-white shadow-md shadow-rose-500/10'
                      : 'bg-slate-950/60 border-slate-800 text-slate-300 hover:border-slate-700'
                  }`}
                >
                  <div className="flex items-center justify-between text-[11px] text-slate-400 mb-1">
                    <span className="font-mono uppercase font-bold text-amber-400">{g.console}</span>
                    <span className="capitalize text-slate-500">{g.genre}</span>
                  </div>
                  <div className="font-semibold text-xs sm:text-sm truncate">{g.name}</div>
                </button>
              ))}
            </div>
          </div>

          {/* Mode & Checkpoint Selection */}
          <div className="flex flex-wrap sm:flex-nowrap gap-3 items-end">
            <div>
              <label className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-2 block flex items-center gap-1.5">
                <Swords className="w-4 h-4 text-indigo-400" />
                Mode
              </label>
              <select
                value={mode}
                onChange={e => {
                  const m = e.target.value as GameMode;
                  setMode(m);
                  resetMatch(selectedGame, m, selectedPresetIdx);
                }}
                className="bg-slate-950 border border-slate-700 text-slate-200 text-xs sm:text-sm rounded-xl px-3 py-2.5 focus:outline-none focus:border-rose-500 font-medium"
              >
                {selectedGame.supportedModes.map(m => (
                  <option key={m} value={m}>
                    {m === 'versus' ? '⚔️ 1v1 Fighting Versus' : m === 'coop' ? '🤝 2-Player Co-Op Partner' : m === 'race' ? '🏎️ Speedrun Race Battle' : '🤖 Autonomous Agent'}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-2 block flex items-center gap-1.5">
                <Cpu className="w-4 h-4 text-emerald-400" />
                AI Checkpoint Difficulty
              </label>
              <select
                value={selectedPresetIdx}
                onChange={e => {
                  const idx = parseInt(e.target.value);
                  setSelectedPresetIdx(idx);
                  resetMatch(selectedGame, mode, idx);
                }}
                className="bg-slate-950 border border-slate-700 text-slate-200 text-xs sm:text-sm rounded-xl px-3 py-2.5 focus:outline-none focus:border-rose-500 font-medium"
              >
                {selectedGame.aiDifficultyPresets.map((p, idx) => (
                  <option key={p.iter} value={idx}>
                    {p.title} ({p.winRateVsHuman}% Win Rate)
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>
      </div>

      {/* Main Interactive Battle Stage & Telemetry Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left 8 Cols: Interactive Canvas & Match Header */}
        <div className="lg:col-span-8 space-y-4">
          <div className="bg-slate-950 border border-slate-800 rounded-2xl overflow-hidden shadow-2xl relative">
            {/* Scoreboard Header (Arcade Style) */}
            <div className="bg-slate-900 border-b border-slate-800 p-4">
              {/* Health Bars & Center Timer */}
              <div className="grid grid-cols-12 items-center gap-3">
                {/* P1 Health */}
                <div className="col-span-5">
                  <div className="flex items-center justify-between text-xs font-bold mb-1">
                    <span className="text-sky-400 flex items-center gap-1">
                      <User className="w-3.5 h-3.5" /> {match.p1Name}
                    </span>
                    <span className="font-mono text-slate-300">{match.p1Health}%</span>
                  </div>
                  <div className="w-full bg-slate-950 rounded-full h-4 p-0.5 border border-sky-500/30 overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all duration-200 bg-gradient-to-r from-sky-500 to-cyan-300"
                      style={{ width: `${match.p1Health}%` }}
                    />
                  </div>
                  <div className="text-[10px] text-slate-400 mt-1 flex justify-between font-mono">
                    <span>SCORE: {match.p1Score.toLocaleString()}</span>
                    <span>WINS: {match.p1Wins}</span>
                  </div>
                </div>

                {/* Center Timer & Round */}
                <div className="col-span-2 text-center">
                  <div className="text-[10px] uppercase font-bold tracking-widest text-slate-400">ROUND {match.currentRound}</div>
                  <div className={`text-2xl sm:text-3xl font-black font-mono tracking-tighter ${
                    match.timer <= 15 ? 'text-rose-500 animate-pulse' : 'text-amber-400'
                  }`}>
                    {match.timer}
                  </div>
                </div>

                {/* P2 Health (AI) */}
                <div className="col-span-5">
                  <div className="flex items-center justify-between text-xs font-bold mb-1">
                    <span className="font-mono text-slate-300">{match.p2Health}%</span>
                    <span className="text-rose-400 flex items-center gap-1">
                      <Cpu className="w-3.5 h-3.5" /> {match.p2Name}
                    </span>
                  </div>
                  <div className="w-full bg-slate-950 rounded-full h-4 p-0.5 border border-rose-500/30 overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all duration-200 bg-gradient-to-l from-rose-500 to-amber-400 ml-auto"
                      style={{ width: `${match.p2Health}%` }}
                    />
                  </div>
                  <div className="text-[10px] text-slate-400 mt-1 flex justify-between font-mono">
                    <span>REACT: {preset.reactionFrames}f</span>
                    <span>SCORE: {match.p2Score.toLocaleString()}</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Canvas Stage */}
            <div className="relative aspect-[16/9] w-full bg-black flex items-center justify-center overflow-hidden">
              <canvas
                ref={canvasRef}
                width={640}
                height={360}
                className="w-full h-full object-contain"
              />

              {/* Watermark / Game Badge */}
              <div className="absolute top-3 left-3 bg-slate-950/80 backdrop-blur-md px-2.5 py-1 rounded-md text-[10px] font-mono text-slate-300 border border-slate-700">
                {selectedGame.name} • {selectedGame.console}
              </div>

              {/* Match Over Banner Overlay */}
              {match.matchOver && (
                <div className="absolute inset-0 bg-black/80 backdrop-blur-sm flex flex-col items-center justify-center p-6 text-center animate-in fade-in zoom-in-95 duration-200">
                  <div className="w-16 h-16 rounded-full bg-amber-500/20 border border-amber-500/50 flex items-center justify-center mb-3">
                    <Trophy className="w-8 h-8 text-amber-400" />
                  </div>
                  <h3 className="text-2xl sm:text-3xl font-black text-white mb-1">
                    {match.winner === 'p1' ? '🎉 YOU DEFEATED THE AI!' : match.winner === 'p2' ? '💀 AI WINS THE MATCH!' : 'DRAW! TIME OVER'}
                  </h3>
                  <p className="text-xs sm:text-sm text-slate-300 max-w-md mb-5">
                    {match.winner === 'p1' 
                      ? `Incredible! You beat the ${preset.title} checkpoint in ${99 - match.timer} seconds. Ready to export the HD video replay?`
                      : `The ${preset.title} exploited your spacing with frame-perfect combo buffers. Try again or step down the iteration difficulty!`}
                  </p>
                  <div className="flex gap-3">
                    <button
                      onClick={() => resetMatch()}
                      className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-white text-xs sm:text-sm font-semibold flex items-center gap-1.5 transition-all"
                    >
                      <RotateCcw className="w-4 h-4" /> Rematch
                    </button>
                    <button
                      onClick={() => {
                        resetMatch();
                        setMatch(prev => ({ ...prev, isPlaying: true }));
                      }}
                      className="px-5 py-2 rounded-xl bg-gradient-to-r from-rose-500 to-amber-500 hover:from-rose-600 hover:to-amber-600 text-white text-xs sm:text-sm font-bold shadow-lg shadow-rose-500/20 flex items-center gap-1.5 transition-all"
                    >
                      <Play className="w-4 h-4" /> Next Round
                    </button>
                  </div>
                </div>
              )}

              {/* Start overlay if idle */}
              {!match.isPlaying && !match.matchOver && (
                <div className="absolute inset-0 bg-slate-950/70 backdrop-blur-xs flex flex-col items-center justify-center p-6 text-center">
                  <div className="w-14 h-14 rounded-full bg-rose-500/20 border border-rose-500/40 flex items-center justify-center mb-3">
                    <Swords className="w-7 h-7 text-rose-400" />
                  </div>
                  <h4 className="text-lg sm:text-xl font-bold text-white mb-1">
                    Human vs. {preset.title}
                  </h4>
                  <p className="text-xs text-slate-300 max-w-md mb-4">
                    Press Start to begin the live interactive match simulator or use your keyboard below to fight in real-time!
                  </p>
                  <button
                    onClick={() => setMatch(prev => ({ ...prev, isPlaying: true }))}
                    className="px-6 py-2.5 rounded-xl bg-gradient-to-r from-rose-500 via-rose-600 to-amber-500 hover:from-rose-600 hover:to-amber-600 text-white font-bold text-sm shadow-xl shadow-rose-600/30 flex items-center gap-2 transition-all transform hover:scale-105"
                  >
                    <Play className="w-4 h-4" /> START MATCH
                  </button>
                </div>
              )}
            </div>

            {/* Bottom Controls Bar */}
            <div className="bg-slate-900 border-t border-slate-800 p-3 flex flex-wrap items-center justify-between gap-3 text-xs">
              <div className="flex items-center space-x-2">
                <button
                  onClick={() => setMatch(prev => ({ ...prev, isPlaying: !prev.isPlaying }))}
                  className={`px-4 py-2 rounded-lg font-bold flex items-center gap-1.5 transition-all ${
                    match.isPlaying
                      ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40 hover:bg-amber-500/30'
                      : 'bg-rose-600 hover:bg-rose-500 text-white'
                  }`}
                >
                  {match.isPlaying ? <><Pause className="w-3.5 h-3.5" /> Pause</> : <><Play className="w-3.5 h-3.5" /> Resume Fight</>}
                </button>
                <button
                  onClick={() => resetMatch()}
                  className="px-3 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 font-medium flex items-center gap-1.5 transition-all"
                >
                  <RotateCcw className="w-3.5 h-3.5" /> Reset
                </button>
              </div>

              {/* Real-time actions display */}
              <div className="flex items-center space-x-4 font-mono text-[11px]">
                <div>
                  <span className="text-slate-400">P1 Action: </span>
                  <span className="text-sky-400 font-bold">{match.p1LastAction}</span>
                </div>
                <div className="h-4 w-px bg-slate-800" />
                <div>
                  <span className="text-slate-400">AI Action: </span>
                  <span className="text-rose-400 font-bold">{match.p2LastAction}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Interactive Keyboard & Arcade Buttons Helper */}
          <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-4">
            <h5 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-3 flex items-center gap-1.5">
              <Gamepad2 className="w-4 h-4 text-sky-400" />
              Player 1 Controls (Interactive Keyboard / Click Buttons)
            </h5>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
              <button
                onMouseDown={() => window.dispatchEvent(new KeyboardEvent('keydown', { key: 'z' }))}
                className={`p-2.5 rounded-xl border text-left transition-all ${
                  activeKeys['z'] ? 'bg-sky-500 border-sky-400 text-white' : 'bg-slate-950/80 border-slate-800 text-slate-300 hover:border-slate-700'
                }`}
              >
                <div className="text-[10px] font-mono text-sky-400 font-bold">KEY [Z] or [J]</div>
                <div className="text-xs font-bold">Standard Punch (B)</div>
              </button>

              <button
                onMouseDown={() => window.dispatchEvent(new KeyboardEvent('keydown', { key: 'x' }))}
                className={`p-2.5 rounded-xl border text-left transition-all ${
                  activeKeys['x'] ? 'bg-sky-500 border-sky-400 text-white' : 'bg-slate-950/80 border-slate-800 text-slate-300 hover:border-slate-700'
                }`}
              >
                <div className="text-[10px] font-mono text-sky-400 font-bold">KEY [X] or [K]</div>
                <div className="text-xs font-bold">Jump / High Kick (A)</div>
              </button>

              <button
                onMouseDown={() => window.dispatchEvent(new KeyboardEvent('keydown', { key: 'c' }))}
                className={`p-2.5 rounded-xl border text-left transition-all ${
                  activeKeys['c'] ? 'bg-amber-500 border-amber-400 text-white' : 'bg-slate-950/80 border-slate-800 text-slate-300 hover:border-slate-700'
                }`}
              >
                <div className="text-[10px] font-mono text-amber-400 font-bold">KEY [C] or [L]</div>
                <div className="text-xs font-bold">Special / Heavy (C)</div>
              </button>

              <button
                onMouseDown={() => window.dispatchEvent(new KeyboardEvent('keydown', { key: 'a' }))}
                className={`p-2.5 rounded-xl border text-left transition-all ${
                  activeKeys['a'] ? 'bg-emerald-500 border-emerald-400 text-white' : 'bg-slate-950/80 border-slate-800 text-slate-300 hover:border-slate-700'
                }`}
              >
                <div className="text-[10px] font-mono text-emerald-400 font-bold">KEY [A] or [S]</div>
                <div className="text-xs font-bold">Guard / Block (Y)</div>
              </button>
            </div>
            <div className="text-[11px] text-slate-500 mt-2 flex items-center justify-between">
              <span>Movement: <strong>Arrow Keys</strong> or <strong>WASD</strong></span>
              <span className="font-mono text-slate-400">USB Gamepad: Auto-mapped in Python script</span>
            </div>
          </div>
        </div>

        {/* Right 4 Cols: AI Neural Telemetry & Terminal Command */}
        <div className="lg:col-span-4 space-y-4">
          {/* AI Neural Decision Visualizer */}
          <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-lg">
            <div className="flex items-center justify-between mb-3">
              <h5 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
                <Activity className="w-4 h-4 text-rose-400" />
                AI Policy Telemetry
              </h5>
              <span className="text-[10px] font-mono bg-slate-950 px-2 py-0.5 rounded text-rose-400 border border-rose-500/20">
                PPO Softmax
              </span>
            </div>

            {/* Simulated 84x84 CNN Visualizer preview */}
            <div className="bg-slate-950 border border-slate-800 rounded-xl p-3 mb-4">
              <div className="flex items-center justify-between text-[11px] text-slate-400 mb-2">
                <span>WarpFrame Input</span>
                <span className="font-mono text-slate-300">84×84 (4 stacked)</span>
              </div>
              <div className="w-full aspect-[4/1] bg-slate-900 rounded border border-slate-800 flex items-center justify-around px-2">
                {[1, 2, 3, 4].map(idx => (
                  <div key={idx} className="w-10 h-10 bg-slate-800 rounded border border-slate-700 flex items-center justify-center">
                    <span className="text-[9px] font-mono text-slate-400">t-{4 - idx}</span>
                  </div>
                ))}
              </div>
              <p className="text-[10px] text-slate-500 mt-1.5 leading-tight">
                {selectedGame.name} frames stacked to compute velocity, jump arc, and attack frames.
              </p>
            </div>

            {/* Action Probability distribution */}
            <div className="space-y-2.5 mb-4">
              <span className="text-[11px] font-semibold text-slate-400 block">Current Action Probabilities:</span>
              {match.aiProbabilities.map((item, idx) => (
                <div key={idx} className="space-y-1">
                  <div className="flex justify-between text-[11px] font-mono">
                    <span className="text-slate-300 font-medium">{item.action}</span>
                    <span className="text-rose-400">{item.prob}%</span>
                  </div>
                  <div className="w-full bg-slate-950 rounded-full h-2 overflow-hidden border border-slate-800">
                    <div
                      className="bg-gradient-to-r from-rose-500 to-amber-400 h-full rounded-full transition-all duration-300"
                      style={{ width: `${item.prob}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>

            {/* Checkpoint Profile Box */}
            <div className="bg-rose-500/10 border border-rose-500/20 rounded-xl p-3 text-xs space-y-1">
              <div className="font-bold text-rose-300 flex items-center gap-1.5">
                <Flame className="w-3.5 h-3.5 text-rose-400" />
                {preset.title}
              </div>
              <p className="text-slate-300 text-[11px] leading-relaxed">
                {preset.description}
              </p>
            </div>
          </div>

          {/* Quick Terminal Launch Command */}
          <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-lg">
            <div className="flex items-center justify-between mb-2">
              <h5 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
                <Terminal className="w-4 h-4 text-emerald-400" />
                Execute in Terminal (WSL2/Linux)
              </h5>
              <button
                onClick={copyCommand}
                className="flex items-center gap-1 px-2.5 py-1 rounded-md bg-slate-800 hover:bg-slate-700 text-[11px] text-slate-200 font-medium transition-all"
              >
                {copiedCmd ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5 text-slate-400" />}
                <span>{copiedCmd ? 'Copied!' : 'Copy'}</span>
              </button>
            </div>

            <div className="bg-slate-950 rounded-xl p-3 border border-slate-800 font-mono text-[11px] text-emerald-400 overflow-x-auto select-all leading-relaxed">
              {generatedCommand}
            </div>

            <p className="text-[10px] text-slate-400 mt-2">
              Runs <code>play_human_vs_ai.py</code> with hardware gamepad/keyboard capture and exports an automatic HD replay MP4.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
