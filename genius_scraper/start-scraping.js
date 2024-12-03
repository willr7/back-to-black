const artistsSongsPage = [
  { artistName: "Drake", artistPath: "Drake" },
  { artistName: "Lil Wayne", artistPath: "Lil-wayne" },
  { artistName: "Tyler, The Creator", artistPath: "Tyler-the-creator" },
  { artistName: "Future", artistPath: "Future" },
  { artistName: "21 Savage", artistPath: "21-savage" },
  { artistName: "J Cole", artistPath: "J-cole" },
  { artistName: "Lil Uzi Vert", artistPath: "Lil-uzi-vert" },
  { artistName: "Lil Yachty", artistPath: "Lil-yachty" },
  { artistName: "Yeat", artistPath: "Yeat" },
  { artistName: "Travis Scott", artistPath: "Travis-scott" },
  { artistName: "Kendrick Lamar", artistPath: "Kendrick-lamar" },
  { artistName: "Playboi Carti", artistPath: "Playboi-carti" },
  { artistName: "Kanye West", artistPath: "Kanye-west" },
  { artistName: "Polo G", artistPath: "Polo-g" },
  { artistName: "A$AP Rocky", artistPath: "A-ap-rocky" },
  { artistName: "Babyface Ray", artistPath: "Babyface-ray" },
  { artistName: "Baby Keem", artistPath: "Baby-keem" },
  { artistName: "Baby Smoove", artistPath: "Baby-smoove" },
  { artistName: "Beyoncé", artistPath: "Beyonce" },
  { artistName: "Big Sean", artistPath: "Big-sean" },
  { artistName: "Boosie Badazz", artistPath: "Boosie-badazz" },
  { artistName: "Busta Rhymes", artistPath: "Busta-rhymes" },
  { artistName: "Bobby Shmurda", artistPath: "Bobby-shmurda" },
  { artistName: "Bryson Tiller", artistPath: "Bryson-tiller" },
  { artistName: "Chief Keef", artistPath: "Chief-keef" },
  { artistName: "Childish Gambino", artistPath: "Childish-gambino" },
  { artistName: "Chris Brown", artistPath: "Chris-brown" },
  { artistName: "Chance the Rapper", artistPath: "Chance-the-rapper" },
  { artistName: "Cardi B", artistPath: "Cardi-b" },
  { artistName: "City Girls", artistPath: "City-girls" },
];

const ENVIRONMENT = "development";

// Create a connection to the background script to keep popup alive
const port = chrome.runtime.connect({ name: "popup" });

// Function to update status in popup and background
function updateStatus(message) {
  const statusDiv = document.getElementById("status");
  if (statusDiv) {
    statusDiv.textContent = message;
  }
  // Also send status to background script
  chrome.runtime.sendMessage({
    type: "updateStatus",
    message: message,
  });
  console.log(message);
}

// Initialize scraping process
async function initializeScraping() {
  if (window.scrapingInProgress) {
    updateStatus("Scraping already in progress...");
    return;
  }

  window.scrapingInProgress = true;
  updateStatus("Starting scraping process...");

  const numArtistsToScrape =
    ENVIRONMENT === "development" ? 5 : artistsSongsPage.length;

  try {
    for (let i = 0; i < numArtistsToScrape; i++) {
      const { artistPath, artistName } = artistsSongsPage[i];
      const artistSongsUrl = `https://genius.com/artists/${artistPath}/songs`;

      updateStatus(
        `Scraping ${artistName}'s lyrics (${i + 1}/${numArtistsToScrape})...`
      );

      // Wait for artist scraping to complete
      await new Promise((resolve) => {
        // First create the tab
        chrome.tabs.create(
          {
            url: artistSongsUrl,
            active: false,
          },
          (tab) => {
            const currentArtistTabId = tab.id;

            // Then set up the message listener for this specific tab
            const messageListener = function (message, sender) {
              if (
                message.type === "artistDone" &&
                sender.tab.id === currentArtistTabId
              ) {
                chrome.runtime.onMessage.removeListener(messageListener);
                resolve();
              }
            };

            chrome.runtime.onMessage.addListener(messageListener);

            // Store current artist info in background script
            chrome.runtime.sendMessage({
              type: "setCurrentArtist",
              artist: {
                artistPath,
                artistName,
                index: i,
                tabId: currentArtistTabId,
              },
            });

            // Notify background script about the new tab
            chrome.runtime.sendMessage({
              type: "artistTabOpened",
              tabId: currentArtistTabId,
            });
          }
        );
      });

      updateStatus(`Finished scraping ${artistName}'s songs`);
    }

    updateStatus("Finished scraping all artists!");
  } catch (error) {
    console.error("Error during scraping:", error);
    updateStatus(`Error: ${error.message}`);
  } finally {
    window.scrapingInProgress = false;
  }
}

// Instead of running in popup, check if we're in the scraper tab
const isScrapeManager = window.location.href.includes("scraper.html");
if (isScrapeManager) {
  document.addEventListener("DOMContentLoaded", function () {
    console.log("Scraper manager loaded");
    document
      .getElementById("scraper-button")
      .addEventListener("click", initializeScraping);
  });
} else {
  // If clicked from popup, open scraper manager in new tab
  document.addEventListener("DOMContentLoaded", function () {
    document
      .getElementById("scraper-button")
      .addEventListener("click", function () {
        chrome.tabs.create({
          url: chrome.runtime.getURL("scraper.html"),
          active: true,
        });
      });
  });
}
