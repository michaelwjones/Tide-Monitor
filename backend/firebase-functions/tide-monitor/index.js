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
        const latest = windData.data[0];
        enrichedData.ws = latest.s || "-999";
        enrichedData.wd = latest.d || "-999"; 
        enrichedData.gs = latest.g || "-999";
      } else {
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
        const latest = waterData.data[0];
        const waterLevel = latest.v || "-999";
        
        // Enhanced validation and logging
        console.log(`NOAA water level data: ${waterLevel} at ${latest.t}, flags: ${latest.f}, quality: ${latest.q}`);
        
        // Check for unusual values (outside expected range 0-8 feet)
        if (waterLevel !== "-999" && (parseFloat(waterLevel) < 0 || parseFloat(waterLevel) > 8)) {
          console.warn(`Unusual NOAA water level detected: ${waterLevel} feet - outside expected range`);
        }
        
        // Check data age (warn if older than 10 minutes)
        const dataTime = new Date(latest.t);
        const now = new Date();
        const ageMinutes = (now - dataTime) / (1000 * 60);
        if (ageMinutes > 10) {
          console.warn(`NOAA water level data is ${ageMinutes.toFixed(1)} minutes old`);
        }
        
        enrichedData.wm = waterLevel;
      } else {
        console.warn('No water level data available from NOAA API');
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
