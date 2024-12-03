async function pause() {
  await new Promise((resolve) => setTimeout(resolve, 1000));
}

const allSongsData = [];

function downloadSong(
  songTitle,
  artistName,
  albumName,
  songLyrics,
  songDate,
  songUrl
) {
  const songData = `Title: ${songTitle}\nArtist: ${artistName}\nAlbum: ${albumName}\nLyrics:\n${songLyrics}\nSong Genius Url:${songUrl}\nSong Date: ${songDate}`;

  downloadText(songData, `${artistName} - ${songTitle}`);
}

function downloadText(text, fileName) {
  const blob = new Blob([text], { type: "text/plain" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${fileName}.txt`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function downloadAllSongs(artistName) {
  const massiveString = allSongsData.join("\n");

  // Generate the zip file and trigger download
  downloadText(massiveString, artistName);
}

function saveSongData(
  songTitle,
  artistName,
  albumName,
  songLyrics,
  songDate,
  songUrl
) {
  const songData = {
    songTitle,
    content: `Title: ${songTitle}\nArtist: ${artistName}\nAlbum: ${albumName}\nLyrics:\n${songLyrics}\nSong Genius Url:${songUrl}\nSong Date: ${songDate}`,
  };
  allSongsData.push(songData); // Save song data to the array
}

function extractTextWithNewlines(node) {
  let text = "";
  node.childNodes.forEach((child) => {
    if (child.nodeName === "BR") {
      text += "\n";
    } else if (child.nodeType === Node.TEXT_NODE) {
      text += child.textContent;
    } else {
      text += extractTextWithNewlines(child);
    }
  });
  return text;
}

async function scrapeSong() {
  if (document.URL.includes("/artists/") && document.URL.includes("/songs")) {
    console.log("Scraping is not allowed on artist songs page.");
    return;
  }

  console.log(`scraping song at url "${document.URL}"...`);

  const songTitle = document.querySelector(
    ".SongHeaderdesktop__HiddenMask-sc-1effuo1-11.iMpFIj"
  )?.textContent;

  const songLyricsContainer = document.querySelector(
    ".Lyrics__Container-sc-1ynbvzw-1.kUgSbL"
  );

  const songLyrics = extractTextWithNewlines(songLyricsContainer);

  const songUrl = document.URL;

  const artistName = document.querySelector(
    ".HeaderArtistAndTracklistdesktop__ListArtists-sc-4vdeb8-1"
  )?.textContent;

  const albumName = (
    document.querySelector(
      ".HeaderArtistAndTracklistdesktop__Tracklist-sc-4vdeb8-2 a"
    ) ?? {}
  )?.textContent;

  const songDate = document.querySelector(
    ".MetadataStats__Container-sc-1t7d8ac-0"
  )?.childNodes[0]?.textContent;

  // saveSongData(songTitle, artistName, albumName, songLyrics, songDate, songUrl);
  downloadSong(songTitle, artistName, albumName, songLyrics, songDate, songUrl);

  // Send completion message through chrome runtime
  console.log("finished downloading song, sending message");
  chrome.runtime.sendMessage({ type: "songDone" });
}

scrapeSong();
