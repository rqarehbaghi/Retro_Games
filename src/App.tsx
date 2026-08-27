import React, { useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Navbar, TabType } from './components/Navbar';
import { MatchArena } from './components/MatchArena';
import { TrainingLab } from './components/TrainingLab';
import { VideoStudio } from './components/VideoStudio';
import { CodeVault } from './components/CodeVault';
import { MonetizationGuide } from './components/MonetizationGuide';

export default function App() {
  const [activeTab, setActiveTab] = useState<TabType>('arena');

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans selection:bg-rose-500 selection:text-white flex flex-col">
      {/* Top Navigation */}
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} />

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.15, ease: 'easeOut' }}
          >
            {activeTab === 'arena' && <MatchArena />}
            {activeTab === 'training' && <TrainingLab />}
            {activeTab === 'video' && <VideoStudio />}
            {activeTab === 'code' && <CodeVault />}
            {activeTab === 'monetization' && <MonetizationGuide />}
          </motion.div>
        </AnimatePresence>
      </main>

      {/* Footer */}
      <footer className="bg-slate-950 border-t border-slate-900 py-6 text-center text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>Retro AI Arena & Automated Video Engine • Stable-Retro + PPO</span>
          <span className="font-mono text-[11px] text-slate-600">
            Pygame 2-Player • WarpFrame (84×84) • VecFrameStack (4) • ffmpeg HD
          </span>
        </div>
      </footer>
    </div>
  );
}
