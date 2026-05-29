'use client';

import React from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';

export default function TabNav() {
  const pathname = usePathname();

  const tabs = [
    { name: 'MLB', href: '/mlb' },
    { name: 'Custom', href: '/custom' },
  ];

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 h-16 bg-gray-900 border-b border-gray-800 flex items-center justify-between px-6 select-none shadow-md backdrop-blur-md bg-opacity-90">
      {/* Logo */}
      <div className="flex items-center space-x-3">
        <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-blue-600 to-red-600 flex items-center justify-center shadow-lg shadow-blue-500/20">
          <span className="text-xs font-bold text-white tracking-wider">PF</span>
        </div>
        <span className="text-lg font-bold tracking-tight text-white bg-clip-text">
          MLB PitchFlow AI
        </span>
      </div>

      {/* Tabs */}
      <div className="flex items-center space-x-2">
        {tabs.map((tab) => {
          const isActive = pathname.startsWith(tab.href);
          return (
            <Link
              key={tab.name}
              href={tab.href}
              className={`px-4 py-2 rounded-lg text-sm font-semibold transition-all duration-300 ${
                isActive
                  ? 'bg-blue-600 text-white shadow-lg shadow-blue-600/30 font-bold'
                  : 'text-gray-400 hover:text-white hover:bg-gray-800'
              }`}
            >
              {tab.name}
            </Link>
          );
        })}
      </div>
    </nav>
  );
}
