import { NextRequest, NextResponse } from 'next/server';
import { PredictResponse } from '../../../../lib/types';

// In-memory store to hold the latest prediction results mapped by gamePk
const latestPrediction: Record<number, PredictResponse> = {};

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { gamePk, ...predictResult } = body;

    if (!gamePk) {
      return NextResponse.json({ error: 'gamePk is required in body' }, { status: 400 });
    }

    // Store in-memory cache
    latestPrediction[Number(gamePk)] = predictResult as PredictResponse;

    console.log(`[Prediction Cached] Successfully stored prediction for gamePk: ${gamePk}`);
    return NextResponse.json({ success: true, message: `Prediction stored for gamePk: ${gamePk}` });
  } catch (error: any) {
    console.error('Error storing live prediction:', error);
    return NextResponse.json(
      { error: error?.message || 'Internal server error during prediction storage.' },
      { status: 500 }
    );
  }
}

export async function GET(req: NextRequest) {
  try {
    const { searchParams } = new URL(req.url);
    const gamePkParam = searchParams.get('gamePk');

    if (!gamePkParam) {
      return NextResponse.json({ error: 'gamePk query parameter is required' }, { status: 400 });
    }

    const gamePk = Number(gamePkParam);
    const prediction = latestPrediction[gamePk] || null;

    return NextResponse.json(prediction);
  } catch (error: any) {
    console.error('Error fetching live prediction:', error);
    return NextResponse.json(
      { error: error?.message || 'Internal server error during prediction retrieval.' },
      { status: 500 }
    );
  }
}
