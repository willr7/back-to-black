// State management
let state = {
  currentArtist: null,
  status: "",
  activeArtistTab: null,
};

// Keep track of active connections
let connections = new Set();

// Message handling
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  //   console.log("Background received message:", message);

  switch (message.type) {
    case "setCurrentArtist":
      state.currentArtist = message.artist;
      break;

    case "artistTabOpened":
      state.activeArtistTab = message.tabId;
      break;

    case "artistDone":
      // Close the artist tab using the sender's tab ID
      if (sender.tab) {
        chrome.tabs.remove(sender.tab.id);
      }
      // Forward the completion message
      chrome.runtime.sendMessage(message);
      break;

    case "updateStatus":
      state.status = message.message;
      break;
  }

  return true;
});

// Handle connections from popup
chrome.runtime.onConnect.addListener((port) => {
  console.log("New connection established");
  connections.add(port);

  port.onDisconnect.addListener(() => {
    connections.delete(port);
  });

  // Send current state to newly connected popup
  if (state.status) {
    port.postMessage({
      type: "statusUpdate",
      message: state.status,
    });
  }
});

// Keep popup alive
setInterval(() => {
  connections.forEach((port) => {
    try {
      port.postMessage({ type: "keepAlive" });
    } catch (e) {
      connections.delete(port);
    }
  });
}, 25000);
