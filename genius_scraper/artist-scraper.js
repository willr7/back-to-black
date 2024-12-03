console.log("scraping page:", document.URL);
// https://genius.com/artists-index/a
// explore writing an AI agent that goes to the lists of artists
//  and uses AI to determine if they speak in AAVE or not
// then parses the ones that are

const ENVIRONMENT = "development";
async function scrape() {
  try {
    await scrollAllSongs();
    await scrapeAndDownloadAllSongs();

    console.log("sending artistDone message...");

    // Try multiple communication methods
    try {
      // Method 1: Try window.opener
      if (window.opener) {
        window.opener.postMessage("artistDone", "*");
      }
      // Method 2: Try parent window
      else if (window.parent !== window) {
        window.parent.postMessage("artistDone", "*");
      }
      // Method 3: Try chrome runtime messaging
      else {
        chrome.runtime.sendMessage({ type: "artistDone" });
      }

      // Close this tab after sending the message
      window.close();
    } catch (error) {
      console.log("Communication error:", error);
    }
  } catch (error) {
    console.error("Error during scraping:", error);
  }
}

function getNumSongs() {
  const artistSongs = document.querySelectorAll(
    ".ListSectiondesktop__Items-sc-53xokv-8.kbIuNQ a"
  );
  return Array.from(artistSongs).length;
}

async function pause() {
  await new Promise((resolve) => setTimeout(resolve, 1000));
}

function getAllSongs() {
  const artistSongs = Array.from(
    document.querySelectorAll(".ListSectiondesktop__Items-sc-53xokv-8.kbIuNQ a")
  );
  const songs = artistSongs.map((ele) => {
    return {
      songUrl: ele.href,
      songName: ele.querySelector("h3").textContent,
      numStreams: ele.querySelector("span").textContent,
    };
  });
  return songs;
}

async function scrapeAndDownloadAllSongs() {
  const songs = getAllSongs();

  const numSongsToScrape = ENVIRONMENT === "development" ? 1 : songs.length;
  for (let i = 0; i < numSongsToScrape; i++) {
    console.log(
      `scraping song ${i + 1} out of ${songs.length}: "${songs[i].songName}"...`
    );
    const songWindow = window.open(songs[i].songUrl);

    // Wait for the new page to load and finish executing
    await new Promise((resolve) => {
      window.addEventListener("message", function onMessage(event) {
        if (event.origin === songWindow.location.origin) {
          // Ensure the message is from the correct origin
          if (event.data === "done") {
            // Check for a specific message indicating completion
            window.removeEventListener("message", onMessage);
            resolve();
          }
        }
      });
    });

    console.log(`finished scraping song "${songs[i].songName}"`);
    songWindow.close();
  }

  console.log("finishing scraping all songs");
}

async function scrollAllSongs() {
  let infiniteLoadSymbol = document.querySelector(
    ".InfiniteScrollSentinel__Container-sc-1c0r21d-0.fXsNrf"
  );

  let previousSongCount = -1;
  let sameSongCount = 0;
  while (sameSongCount <= 3 && ENVIRONMENT !== "development") {
    console.log(
      `${getNumSongs()} songs loaded. scrolling more songs into view...`
    );

    // scroll down to load symbol
    infiniteLoadSymbol.scrollIntoView({ behavior: "smooth" });

    // wait a little
    await pause();

    // scroll up a little bit
    const closeToPageBottom = window.scrollY - 800;
    window.scrollTo({
      top: closeToPageBottom,
    });

    // wait a little
    await pause();

    infiniteLoadSymbol = document.querySelector(
      ".InfiniteScrollSentinel__Container-sc-1c0r21d-0.fXsNrf"
    );

    if (previousSongCount === getNumSongs()) {
      sameSongCount += 1;
    } else {
      sameSongCount = 0; // set this back to 0
    }
    previousSongCount = getNumSongs();
  }

  console.log("done scrolling!");
  console.log("songs:", getAllSongs());
}

scrape();
