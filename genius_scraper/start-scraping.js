const artistsSongsPage = [
  { artistName: "Drake", artistPath: "Drake" },
  // { artistName: "Lil Wayne", artistPath: "Lil-wayne" },
  // { artistName: "Tyler, The Creator", artistPath: "Tyler-the-creator" },
  // { artistName: "Future", artistPath: "Future" },
  // { artistName: "21 Savage", artistPath: "21-savage" },
  // { artistName: "J Cole", artistPath: "J-cole" },
  // { artistName: "Lil Uzi Vert", artistPath: "Lil-uzi-vert" },
  // { artistName: "Lil Yachty", artistPath: "Lil-yachty" },
  // { artistName: "Yeat", artistPath: "Yeat" },
  // { artistName: "Travis Scott", artistPath: "Travis-scott" },
  // { artistName: "Kendrick Lamar", artistPath: "Kendrick-lamar" },
  // { artistName: "Playboi Carti", artistPath: "Playboi-carti" },
  // { artistName: "Kanye West", artistPath: "Kanye-west" },
  // { artistName: "Polo G", artistPath: "Polo-g" },
  // { artistName: "A$AP Rocky", artistPath: "A-ap-rocky" },
  // { artistName: "Babyface Ray", artistPath: "Babyface-ray" },
  // { artistName: "Baby Keem", artistPath: "Baby-keem" },
  // { artistName: "Baby Smoove", artistPath: "Baby-smoove" },
  // { artistName: "Beyoncé", artistPath: "Beyonce" },
  // { artistName: "Big Sean", artistPath: "Big-sean" },
  // { artistName: "Boosie Badazz", artistPath: "Boosie-badazz" },
  // { artistName: "Busta Rhymes", artistPath: "Busta-rhymes" },
  // { artistName: "Bobby Shmurda", artistPath: "Bobby-shmurda" },
  // { artistName: "Bryson Tiller", artistPath: "Bryson-tiller" },
  // { artistName: "Chief Keef", artistPath: "Chief-keef" },
  // { artistName: "Childish Gambino", artistPath: "Childish-gambino" },
  // { artistName: "Chris Brown", artistPath: "Chris-brown" },
  // { artistName: "Chance the Rapper", artistPath: "Chance-the-rapper" },
  // { artistName: "Cardi B", artistPath: "Cardi-b" },
  // { artistName: "City Girls", artistPath: "City-girls" },
];

document.addEventListener("DOMContentLoaded", function () {
  document
    .getElementById("scraper-button")
    .addEventListener("click", function () {
      console.log("scraping lyrics...");
      alert("scraping lyrics...");
      artistsSongsPage.forEach(async ({ artistPath, artistName }) => {
        const artistSongsUrl = `https://genius.com/artists/${artistPath}/songs`;

        console.log(
          `scraping ${artistName}'s lyrics at "${artistSongsUrl}..."`
        );
        const newWindow = window.open(artistSongsUrl);

        // Wait for the new page to load and finish executing
        await new Promise((resolve) => {
          window.addEventListener("message", function onMessage(event) {
            if (event.origin === newWindow.location.origin) {
              // Ensure the message is from the correct origin
              if (event.data === "done") {
                // Check for a specific message indicating completion
                window.removeEventListener("message", onMessage);
                resolve();
              }
            }
          });
        });

        console.log("finished scraping all artists songs");
      });
    });
});
