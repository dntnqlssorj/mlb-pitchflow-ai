import { NextRequest, NextResponse } from 'next/server';

const PITCH_CODE_MAP: Record<string, string | null> = {
  'FF': 'FF',   // Four-Seam Fastball
  'FT': 'FT',   // Two-Seam Fastball
  'SI': 'SI',   // Sinker
  'SL': 'SL',   // Slider
  'CH': 'CH',   // Changeup
  'CU': 'CU',   // Curveball
  'KC': 'KC',   // Knuckle Curve
  'CB': 'CB',   // Curveball (alt)
  'FC': 'FC',   // Cutter
  'FS': 'FS',   // Splitter
  'KN': 'KN',   // Knuckleball
  'ST': 'SL',   // Sweeper -> SL
  'SV': 'SL',   // Slurve -> SL
  'SC': null,   // Screwball -> ignore
  'EP': null,   // Eephus -> ignore
  'FO': null,   // Forkball -> ignore
};

export async function GET(req: NextRequest) {
  try {
    const { searchParams } = new URL(req.url);
    const playerId = searchParams.get('playerId');

    if (!playerId) {
      return NextResponse.json({ arsenal: ['FF', 'SL', 'CH'] });
    }

    const mlbApiUrl = `https://statsapi.mlb.com/api/v1/people/${playerId}?hydrate=stats(group=pitching,type=pitchArsenal,season=2025)&season=2025`;

    const response = await fetch(mlbApiUrl, {
      signal: AbortSignal.timeout(10000),
    });

    if (!response.ok) {
      return NextResponse.json({ arsenal: ['FF', 'SL', 'CH'] });
    }

    const data = await response.json();
    const person = data.people?.[0];
    
    if (!person || !person.stats) {
      return NextResponse.json({ arsenal: ['FF', 'SL', 'CH'] });
    }

    // Find stats where type.displayName is 'pitchArsenal'
    const arsenalStat = person.stats.find(
      (stat: any) => stat.type?.displayName === 'pitchArsenal'
    );

    if (!arsenalStat || !arsenalStat.splits) {
      return NextResponse.json({ arsenal: ['FF', 'SL', 'CH'] });
    }

    const extractedPitches = new Set<string>();
    
    arsenalStat.splits.forEach((split: any) => {
      const mlbCode = split.stat?.type?.code;
      if (mlbCode) {
        const mappedCode = PITCH_CODE_MAP[mlbCode];
        if (mappedCode) {
          extractedPitches.add(mappedCode);
        }
      }
    });

    const finalArsenal = Array.from(extractedPitches);

    // Use fallback if no pitches are identified (e.g. batters, freshmen)
    return NextResponse.json({
      arsenal: finalArsenal.length > 0 ? finalArsenal : ['FF', 'SL', 'CH'],
    });
  } catch (error: any) {
    console.error('Pitcher arsenal API error:', error);
    // Silent recovery with fallback
    return NextResponse.json({ arsenal: ['FF', 'SL', 'CH'] });
  }
}
