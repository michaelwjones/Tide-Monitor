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
        enrichedData.wm = latest.v || "-999";
      } else {
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
