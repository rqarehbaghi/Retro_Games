import React from 'react';
import { Swords, FlaskConical, Video, Terminal, TrendingUp, Sparkles } from 'lucide-react';

export type TabType = 'arena' | 'training' | 'video' | 'code' | 'monetization';

interface NavbarProps {
  activeTab: TabType;
  setActiveTab: (tab: TabType) => void;
  isAiThinking?: boolean;
}

export const Navbar: React.FC<NavbarProps> = ({ activeTab, setActiveTab, isAiThinking }) => {
  const tabs = [
    { id: 'arena' as TabType, label: 'Match Arena (1v1 & Co-Op)', icon: Swords, badge: 'LIVE' },
    { id: 'training' as TabType, label: 'Training Lab & Checkpoints', icon: FlaskConical },
    { id: 'video' as TabType, label: 'Video Studio & Reels', icon: Video, badge: 'YouTube/Shorts' },
    { id: 'code' as TabType, label: 'Python Scripts & CLI', icon: Terminal },
    { id: 'monetization' as TabType, label: 'Growth & Money Strategy', icon: TrendingUp },
  ];

  return (
    <header className="sticky top-0 z-50 bg-slate-950/90 backdrop-blur-md border-b border-slate-800 text-slate-100">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo & Branding */}
          <div className="flex items-center space-x-3 cursor-pointer" onClick={() => setActiveTab('arena')}>
            <div className="w-10 h-10 rounded-lg bg-gradient-to-tr from-amber-500 via-rose-600 to-indigo-600 flex items-center justify-center shadow-lg shadow-rose-600/20 border border-rose-400/30">
              <Swords className="w-5 h-5 text-white" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="font-black text-lg tracking-tight bg-gradient-to-r from-amber-400 via-rose-400 to-indigo-300 bg-clip-text text-transparent">
                  RETRO AI ARENA
                </span>
                <span className="text-[10px] uppercase font-bold tracking-widest px-1.5 py-0.5 rounded bg-rose-500/20 text-rose-400 border border-rose-500/30">
                  PPO 2-Player
                </span>
              </div>
              <p className="text-xs text-slate-400 hidden sm:block">
                Human vs. AI Gameplay & Automated Content Engine
              </p>
            </div>
          </div>

          {/* Navigation Tabs */}
          <nav className="flex items-center space-x-1 sm:space-x-2 overflow-x-auto py-1">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`relative flex items-center space-x-2 px-3.5 py-2 rounded-lg text-xs sm:text-sm font-medium transition-all duration-150 whitespace-nowrap ${
                    isActive
                      ? 'bg-gradient-to-r from-rose-500/20 to-indigo-500/20 text-white border border-rose-500/40 shadow-sm'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/80 border border-transparent'
                  }`}
                >
                  <Icon className={`w-4 h-4 ${isActive ? 'text-rose-400' : 'text-slate-400'}`} />
                  <span>{tab.label}</span>
                  {tab.badge && (
                    <span className={`text-[10px] px-1.5 py-0.2 rounded-full font-bold uppercase ${
                      isActive ? 'bg-rose-500 text-white' : 'bg-slate-800 text-slate-300'
                    }`}>
                      {tab.badge}
                    </span>
                  )}
                </button>
              );
            })}
          </nav>
        </div>
      </div>
    </header>
  );
};
