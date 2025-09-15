# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Tide Monitor** - Ultrasonic sensor system with Particle Boron 404X that measures water levels and wave heights, automatically enriched with NOAA environmental data via Firebase Cloud Functions.

**Components**: Embedded firmware → Web dashboards → Firebase Cloud Functions (enrichment, LSTM/transformer forecasting, tidal analysis)

**Data flow**: `Sensor → Particle → Firebase → Cloud Functions → Enriched Data → Dashboard`

## Documentation Structure

**For detailed information, consult the README.md file at the appropriate scope level:**
- `/README.md` - Project overview and getting started
- `/backend/README.md` - Backend architecture  
- `/backend/firebase-functions/README.md` - Cloud Functions overview
- `/backend/firebase-functions/tidal-analysis/README.md` - Analysis methods
- Method-specific README files (lstm/v1, transformer/v1, etc.) - Implementation details

This CLAUDE.md contains only essential development guidance. All feature details, technical specifications, and architectural information are in the hierarchical README files.

## Essential Development Commands

1. **Firmware**: `flash.bat` in `backend/boron404x/` to deploy to Particle Boron
2. **Dashboards**: Open `index.html` or `debug/index.html` directly in browser  
3. **Firebase Functions**: 
   - **Safe deploy**: `deploy-enrichment.bat` (NOAA enrichment)
   - **Analysis deploy**: `deploy-matrix-pencil-v1.bat`, `deploy-lstm-v1.bat`, `deploy-transformer-v1.bat`
   - **Prerequisites**: `npm install` in each function directory before first deployment
   - **Logs**: `firebase functions:log`
4. **Model Testing**: Use testing interfaces in respective function folders (see README files)

## Critical Technical Rules

**Firebase Data Schema**: Sensor data (t, w, hp, he, wp, we, vs) + NOAA data (ws, wd, gs, wm). NOAA fields = -999 when APIs fail.

**Firebase endpoint**: `https://tide-monitor-boron-default-rtdb.firebaseio.com/readings/`

**Transformer v1 CRITICAL**: Always use single-pass encoder-only architecture (433→144 mapping). Never implement autoregressive generation.

**Data validation**: NOAA APIs require exactly 1 data point with expected fields (wind: s,d,g; water: v,t).

**Removed**: Binning analysis method (hb/wb fields) - removed due to performance issues.

## Memory & Context

Always keep these rules in your context:

- Do not mask a failure with a fallback
- Occasionally I will ask for options, ideas, or advice. In those cases, do not make any code changes.
- Follow directions exactly, make suggestions separately, don't make unauthorized changes.
- When I ask you to make a change A, do not make changes A and B. Only change A. I am a very technical user, if I want B changed I will ask for it.
- Feel free to make suggestions if you think they will help me arrive at my goal.
- Don't run a firebase command unless specifically asked to.
- README files should contain current state, not change history.
- Do not put emojis or unicode in batch files or powershell scripts.
- Do not ever try to pass off fake data as real data.