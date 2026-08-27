import React from 'react';
import { 
  TrendingUp, DollarSign, Users, Eye, Sparkles, Trophy, 
  Flame, Target, ArrowUpRight, ShieldCheck, Zap, Layers 
} from 'lucide-react';

export const MonetizationGuide: React.FC = () => {
  const revenueStreams = [
    {
      title: 'YouTube Long-Form AdSense',
      potential: '$1,500 - $8,000 / mo',
      difficulty: 'Medium',
      description: '8-15 minute narrative videos breaking down training milestones, agent mistakes, and the climactic Human vs. AI final showdown.',
      tactics: ['Place mid-rolls at iteration milestones', 'Structure videos with 3-act tension', 'Engage audience in comment polls for next game']
    },
    {
      title: 'Retro Handheld & Controller Affiliates',
      potential: '$500 - $3,500 / mo',
      difficulty: 'Easy',
      description: 'Retro gamers love physical hardware. Link recommended 8BitDo arcade sticks, Anbernic/Miyoo handhelds, and CRT scalers in descriptions.',
      tactics: ['"The exact arcade stick I used to fight the AI"', 'Amazon & Retro hardware store affiliate tags', 'High conversion from nostalgic viewers']
    },
    {
      title: 'Cloud GPU & ML Tool Sponsors',
      potential: '$2,000 - $10,000 / deal',
      difficulty: 'Medium-High',
      description: 'AI cloud compute providers (RunPod, Lambda Labs, Vast.ai) actively sponsor developers demonstrating scalable machine learning in production.',
      tactics: ['Show how fast PPO trains on cloud GPUs', 'Dedicated 60s sponsor segment', 'Custom sign-up referral promo codes']
    },
    {
      title: 'Patreon & Model Checkpoint Vault',
      potential: '$800 - $4,000 / mo',
      difficulty: 'Easy',
      description: 'Offer supporters access to raw .zip checkpoint weights, custom scenario.json configs, pre-configured WSL2 Docker images, and voting on next games.',
      tactics: ['Tier 1 ($5): Download all AI Model weights', 'Tier 2 ($15): Vote on the next game to train', 'Tier 3 ($50): AI Coaching / Custom integration']
    }
  ];

  const viralFormats = [
    {
      format: 'Human vs. My Own AI: The 30-Day Showdown',
      hook: 'Trained a neural net on Mortal Kombat / Street Fighter from zero knowledge, then sat down on Player 1 with a real controller to see if I can win 1 round.',
      whyItWorks: 'Creates genuine human vs machine stakes. Viewers root for the creator while marveling at the AI frame-perfect counters.',
      retentionScore: '94% AVD'
    },
    {
      format: 'Can an AI Co-Op Partner Carry Me Through Contra?',
      hook: 'I trained Player 2 to be an aimbot in Contra. My challenge: I cannot shoot, I can only dodge. Can the AI beat the final boss alone?',
      whyItWorks: 'Hilarious dynamic. The AI will inevitably steal powerups or make chaotic plays, creating comedy and suspense.',
      retentionScore: '91% AVD'
    },
    {
      format: 'Speedrun Ghost Battle: Human vs. AI TAS',
      hook: 'Split-screen side-by-side race in Mario 3 or Sonic 2. Human speedrunner splits on left vs AI model on right.',
      whyItWorks: 'High visual comparison appeal for both casuals and speedrun enthusiasts.',
      retentionScore: '88% AVD'
    }
  ];

  return (
    <div className="space-y-6">
      {/* Top Banner */}
      <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-xl backdrop-blur-sm">
        <div className="flex items-center space-x-2">
          <TrendingUp className="w-5 h-5 text-emerald-400" />
          <h2 className="text-lg font-bold text-white">Audience Growth & Monetization Blueprint</h2>
        </div>
        <p className="text-xs text-slate-400 mt-1">
          How to turn your Reinforcement Learning retro pipeline into viral reach, engaged followers, and diverse revenue streams.
        </p>
      </div>

      {/* 3-Step Repurposing Engine */}
      <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-lg">
        <h3 className="text-sm font-bold uppercase tracking-wider text-slate-300 mb-4 flex items-center gap-1.5">
          <Flame className="w-4 h-4 text-amber-400" />
          The 1-Run $\rightarrow$ 4-Asset Multiplier Engine
        </h3>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-slate-950 border border-slate-800 rounded-xl p-4 space-y-2">
            <div className="w-8 h-8 rounded-lg bg-rose-500/20 text-rose-400 flex items-center justify-center font-black text-xs">
              01
            </div>
            <h4 className="font-bold text-slate-200 text-xs sm:text-sm">9:16 Shorts & TikToks</h4>
            <p className="text-[11px] text-slate-400 leading-relaxed">
              15s progression clips (Iter 1 vs Iter 500). Primary engine for rapid subscriber and follower acquisition.
            </p>
          </div>

          <div className="bg-slate-950 border border-slate-800 rounded-xl p-4 space-y-2">
            <div className="w-8 h-8 rounded-lg bg-indigo-500/20 text-indigo-400 flex items-center justify-center font-black text-xs">
              02
            </div>
            <h4 className="font-bold text-slate-200 text-xs sm:text-sm">10-Min YouTube Story</h4>
            <p className="text-[11px] text-slate-400 leading-relaxed">
              Deep dive into the struggle, reward curve, funny bugs, and live Human vs AI match. Primary AdSense revenue.
            </p>
          </div>

          <div className="bg-slate-950 border border-slate-800 rounded-xl p-4 space-y-2">
            <div className="w-8 h-8 rounded-lg bg-emerald-500/20 text-emerald-400 flex items-center justify-center font-black text-xs">
              03
            </div>
            <h4 className="font-bold text-slate-200 text-xs sm:text-sm">GitHub & Model Hub</h4>
            <p className="text-[11px] text-slate-400 leading-relaxed">
              Open source checkpoint weights (.zip) on Hugging Face / GitHub. Drives technical authority, stars, and backlinks.
            </p>
          </div>

          <div className="bg-slate-950 border border-slate-800 rounded-xl p-4 space-y-2">
            <div className="w-8 h-8 rounded-lg bg-amber-500/20 text-amber-400 flex items-center justify-center font-black text-xs">
              04
            </div>
            <h4 className="font-bold text-slate-200 text-xs sm:text-sm">Patreon & Hardware</h4>
            <p className="text-[11px] text-slate-400 leading-relaxed">
              Affiliate links for retro controllers + Patreon voting on the next game to conquer.
            </p>
          </div>
        </div>
      </div>

      {/* High-Retention Challenge Formats */}
      <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-lg">
        <h3 className="text-sm font-bold uppercase tracking-wider text-slate-300 mb-3 flex items-center gap-1.5">
          <Trophy className="w-4 h-4 text-rose-400" />
          High-Retention Content Concepts
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {viralFormats.map((item, idx) => (
            <div key={idx} className="bg-slate-950 border border-slate-800 rounded-xl p-4 space-y-2 flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-[10px] uppercase font-bold font-mono px-2 py-0.5 rounded bg-rose-500/20 text-rose-300">
                    {item.retentionScore}
                  </span>
                </div>
                <h4 className="font-bold text-slate-200 text-xs sm:text-sm mb-1">{item.format}</h4>
                <p className="text-[11px] text-slate-400 leading-relaxed mb-2">
                  <strong>The Setup:</strong> {item.hook}
                </p>
              </div>
              <p className="text-[10px] text-emerald-400 bg-emerald-500/10 p-2 rounded border border-emerald-500/20">
                💡 {item.whyItWorks}
              </p>
            </div>
          ))}
        </div>
      </div>

      {/* Revenue Streams Grid */}
      <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-lg">
        <h3 className="text-sm font-bold uppercase tracking-wider text-slate-300 mb-4 flex items-center gap-1.5">
          <DollarSign className="w-4 h-4 text-emerald-400" />
          Monetization Streams Breakdown
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {revenueStreams.map((stream, idx) => (
            <div key={idx} className="bg-slate-950 border border-slate-800 rounded-xl p-4 space-y-2">
              <div className="flex items-center justify-between">
                <h4 className="font-bold text-slate-200 text-sm">{stream.title}</h4>
                <span className="text-xs font-mono font-bold text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">
                  {stream.potential}
                </span>
              </div>
              <p className="text-[11px] text-slate-400 leading-relaxed">{stream.description}</p>
              <div className="pt-2 border-t border-slate-800/80 space-y-1">
                <span className="text-[10px] font-bold text-slate-500 uppercase">Key Execution Tactics:</span>
                {stream.tactics.map((t, tIdx) => (
                  <div key={tIdx} className="text-[11px] text-slate-300 flex items-center gap-1.5">
                    <span className="w-1 h-1 rounded-full bg-emerald-400" />
                    <span>{t}</span>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
