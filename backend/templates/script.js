
        document.getElementById("listen-btn").addEventListener("click", function() {
            const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
            recognition.lang = "en-US";

            recognition.onstart = function() {
                document.getElementById("recognized-text").innerHTML = "🎤 Listening...";
            };

            recognition.onspeechend = function() {
                recognition.stop();
            };

            recognition.onresult = function(event) {
                let sentence = event.results[0][0].transcript.toLowerCase();
                document.getElementById("recognized-text").innerHTML = `Recognized: <span>${sentence}</span>`;
                getTranslation(sentence);
            };

            recognition.onerror = function(event) {
                document.getElementById("recognized-text").innerHTML = "❌ Error: " + event.error;
            };

            recognition.start();
        });

        async function getTranslation(sentence) {
            const response = await fetch("/translate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ sentence })
            });

            const data = await response.json();
            playVideos(data.videos);
        }

        function playVideos(videoList) {
            let videoElement = document.getElementById("signVideo");
            let index = 0;

            function playNext() {
                if (index < videoList.length) {
                    videoElement.src = videoList[index];
                    videoElement.play();
                    index++;
                }
            }

            videoElement.onended = playNext;
            playNext();
        }
   