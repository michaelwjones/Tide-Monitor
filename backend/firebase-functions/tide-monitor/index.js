const {onValueCreated} = require("firebase-functions/v2/database");
const {initializeApp} = require("firebase-admin/app");
const {getDatabase} = require("firebase-admin/database");
const https = require("https");

initializeApp();

async function fetchNoaaData(url) {
  return new Promise((resolve, reject) => {
    https.get(url, (res) => {
      let data = '';
      res.on('data', (chunk) => {
        data += chunk;
      });
      res.on('end', () => {
        try {
          const parsed = JSON.parse(data);
          resolve(parsed);
        } catch (error) {
          reject(error);
        }
      });
    }).on('error', (error) => {
      reject(error);
    });
  });
}

exports.enrichTideData = onValueCreated("/readings/{readingId}", async (event) => {
    const snapshot = event.data;
    const readingId = event.params.readingId;
    const reading = snapshot.val();
    
    console.log(`Processing new reading: ${readingId}`);
    
    const enrichedData = { ...reading };
    
    try {
      const windUrl = 'https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?date=latest&station=8656483&product=wind&units=metric&time_zone=lst_ldt&format=json&application=Michael.wayne.jones@gmail.com';
      const windData = await fetchNoaaData(windUrl);
      
      if (windData.data && windData.data.length > 0) {
        // Use the last element because NOAA API occasionally returns the whole day's data instead of just the latest reading
        const latest = windData.data[windData.data.length - 1];
        if (latest.s && latest.d && latest.g) {
          enrichedData.ws = latest.s;
          enrichedData.wd = latest.d; 
          enrichedData.gs = latest.g;
        } else {
          console.error('Wind data missing expected fields:', { s: latest.s, d: latest.d, g: latest.g });
          enrichedData.ws = "-999";
          enrichedData.wd = "-999";
          enrichedData.gs = "-999";
        }
      } else {
        console.error('Wind data array is empty or missing:', windData.data?.length || 0);
        enrichedData.ws = "-999";
        enrichedData.wd = "-999";
        enrichedData.gs = "-999";
      }
    } catch (error) {
      console.error('Wind data fetch failed:', error);
      enrichedData.ws = "-999";
      enrichedData.wd = "-999";
      enrichedData.gs = "-999";
    }
    
    try {
      const waterUrl = 'https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?date=latest&station=8656483&product=water_level&datum=MLLW&time_zone=lst_ldt&units=english&format=json&application=Michael.wayne.jones@gmail.com';
      const waterData = await fetchNoaaData(waterUrl);
      
      if (waterData.data && waterData.data.length > 0) {
        // Use the last element because NOAA API occasionally returns the whole day's data instead of just the latest reading
        const latest = waterData.data[waterData.data.length - 1];
        if (latest.v && latest.t) {
          const waterLevel = latest.v;
          
          // Enhanced logging for troubleshooting discontinuous values
          console.log(`NOAA water level data: ${waterLevel} at ${latest.t}, flags: ${latest.f}, quality: ${latest.q}`);
          
          // Check data age (warn if older than 10 minutes)
          // NOAA API returns time in Eastern Time (EST/EDT) due to time_zone=lst_ldt
          // Convert to UTC for proper comparison with Firebase Functions time
          const dataTimeString = latest.t; // e.g., "2025-07-16 16:00"
          
          // Parse the Eastern Time and convert to UTC
          const [datePart, timePart] = dataTimeString.split(' ');
          const [year, month, day] = datePart.split('-').map(Number);
          const [hour, minute] = timePart.split(':').map(Number);
          
          // Create date in local Eastern Time, then convert to UTC
          // July is EDT (UTC-4), so we need to add 4 hours to get UTC
          const dataTime = new Date(year, month - 1, day, hour, minute);
          const now = new Date();
          
          // Assume EDT for now (UTC-4) - could be enhanced to detect EST/EDT
          const dataTimeUTC = new Date(dataTime.getTime() + (4 * 60 * 60 * 1000));
          const ageMinutes = (now - dataTimeUTC) / (1000 * 60);
          
          if (ageMinutes > 10) {
            console.warn(`NOAA water level data is ${ageMinutes.toFixed(1)} minutes old`);
          }
          
          enrichedData.wm = waterLevel;
        } else {
          console.error('Water level data missing expected fields:', { v: latest.v, t: latest.t });
          enrichedData.wm = "-999";
        }
      } else {
        console.error('Water level data array is empty or missing:', waterData.data?.length || 0);
        enrichedData.wm = "-999";
      }
    } catch (error) {
      console.error('Water level data fetch failed:', error);
      enrichedData.wm = "-999";
    }
    
    const db = getDatabase();
    await db.ref(`/readings/${readingId}`).set(enrichedData);
    console.log(`Enriched reading ${readingId} with NOAA data`);
    
    return null;
  });
