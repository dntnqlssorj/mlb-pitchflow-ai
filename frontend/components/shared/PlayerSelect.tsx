'use client';

import React, { useState, useEffect, useRef } from 'react';

interface SearchResult {
  id: number;
  fullName: string;
  teamId: string;
  teamName: string;
  position: string;
  pitchHand?: 'R' | 'L';
  batSide?: 'R' | 'L';
}

interface PlayerSelectProps {
  label: string;
  onSelect: (playerId: number, teamId: string, teamName: string, hand: 'R' | 'L') => void;
  excludeId?: number;
}

export default function PlayerSelect({ label, onSelect, excludeId }: PlayerSelectProps) {
  const [query, setQuery] = useState<string>('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [isSearching, setIsSearching] = useState<boolean>(false);
  const [selected, setSelected] = useState<SearchResult | null>(null);
  const [isOpen, setIsOpen] = useState<boolean>(false);
  
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Debounced search logic
  useEffect(() => {
    if (query.trim().length < 2) {
      setResults([]);
      setIsSearching(false);
      return;
    }

    // Do not search if the query perfectly matches the currently selected player's name
    if (selected && query === selected.fullName) {
      return;
    }

    setIsSearching(true);
    const delayDebounceFn = setTimeout(async () => {
      try {
        const res = await fetch(`/api/player-search?query=${encodeURIComponent(query)}`);
        if (res.ok) {
          const data = await res.json();
          let people: SearchResult[] = data.people || [];
          
          // Filter out the duplicate opposing role
          if (excludeId !== undefined) {
            people = people.filter((p) => p.id !== excludeId);
          }
          
          setResults(people);
          setIsOpen(true);
        }
      } catch (err) {
        console.error('Error searching players:', err);
      } finally {
        setIsSearching(false);
      }
    }, 300);

    return () => clearTimeout(delayDebounceFn);
  }, [query, excludeId, selected]);

  // Close dropdown on click outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value;
    setQuery(val);
    
    // Clear selection if input is edited
    if (selected && val !== selected.fullName) {
      setSelected(null);
    }
  };

  const handleSelectPlayer = (player: SearchResult) => {
    setSelected(player);
    setQuery(player.fullName);
    setResults([]);
    setIsOpen(false);
    
    // Resolve R/L handedness depending on player label context
    const hand = label.includes('타자')
      ? (player.batSide || 'R')
      : (player.pitchHand || 'R');

    onSelect(player.id, player.teamId, player.teamName, hand as 'R' | 'L');
  };

  return (
    <div className="bg-gray-800/40 rounded-xl p-4 border border-gray-700/50 space-y-2.5 relative" ref={dropdownRef}>
      <div className="flex items-center justify-between">
        <label className="text-sm font-bold text-gray-300 tracking-wide uppercase">
          {label} 검색
        </label>
        {excludeId && selected && excludeId === selected.id && (
          <span className="text-xs font-semibold text-red-500 animate-pulse">
            동일한 선수 선택 불가
          </span>
        )}
      </div>

      <div className="relative">
        <input
          type="text"
          value={query}
          onChange={handleInputChange}
          onFocus={() => {
            if (results.length > 0) setIsOpen(true);
          }}
          placeholder="선수 이름 검색 (예: Gerrit Cole)"
          className={`bg-gray-800 border rounded px-3 py-2 w-full text-white placeholder-gray-500 focus:outline-none transition duration-200 pr-10 ${
            selected
              ? 'border-green-500/80 focus:ring-2 focus:ring-green-500/30'
              : 'border-gray-600 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20'
          }`}
        />

        {/* Input indicators (Spinner / Green Checkmark) */}
        <div className="absolute right-3 top-1/2 -translate-y-1/2 flex items-center justify-center pointer-events-none">
          {isSearching && (
            <svg
              className="animate-spin h-4 w-4 text-blue-400"
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
            >
              <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="4"
              ></circle>
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
              ></path>
            </svg>
          )}
          {!isSearching && selected && (
            <svg
              className="h-5 w-5 text-green-500"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.5"
              viewBox="0 0 24 24"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7"></path>
            </svg>
          )}
        </div>
      </div>

      {/* Autocomplete Dropdown overlay */}
      {isOpen && results.length > 0 && (
        <div className="absolute left-0 right-0 z-50 mt-1 mx-4 bg-gray-900 border border-gray-700 rounded-lg max-h-48 overflow-y-auto shadow-2xl divide-y divide-gray-800 scrollbar-thin scrollbar-thumb-gray-800">
          {results.map((player) => (
            <div
              key={player.id}
              onClick={() => handleSelectPlayer(player)}
              className="px-3.5 py-2.5 text-xs text-gray-200 hover:bg-gray-850 hover:text-white cursor-pointer transition flex items-center justify-between"
            >
              <div className="font-semibold">{player.fullName}</div>
              <div className="text-gray-500 text-[10px]">
                {player.position} · {player.teamName}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
