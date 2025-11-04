window.onload = function() {
  console.log("Page loaded. Initializing experiment...");

  const jsPsych = initJsPsych({
    on_finish: function() {
      jsPsych.data.displayData("json");
    }
  });

  const instructions = {
    type: jsPsychInstructions,
    pages: [
      'Welcome to the experiment. Click next to begin.',
      'In this experiment, sentences will be presented word by word. After each sentence, you will answer a question.',
    ],
    show_clickable_nav: true
  };

  // Function to load and parse the CSV file
  function loadStimuliFromCSV(filePath) {
    return new Promise((resolve, reject) => {
      console.log(`Attempting to load stimuli from ${filePath}`);
      Papa.parse(filePath, {
        download: true,
        header: false, // Assuming the CSV does not have a header row
        complete: function(results) {
          console.log("Stimuli loaded successfully.");
          resolve(results.data);
        },
        error: function(error) {
          console.error("Error loading stimuli: ", error);
          reject(error);
        }
      });
    });
  }

  // Function to create RSVP trials from a sentence
  function createRSVPTrials(sentence, prompt) {
    let trials = [];
    const words = sentence.trim().split(/\s+/); // Split by any whitespace

    if ((sentence != "") & (sentence != "Sentence")) {
      console.log(`Creating RSVP trials for sentence: "${sentence}" with prompt: "${prompt}"`);

      // Display each word for 200ms, with a SOA of 366ms
      words.forEach((word, index) => {
        trials.push({
          type: jsPsychHtmlKeyboardResponse,
          stimulus: `<p style="font-size: 32px;">${word}</p>`,
          choices: "NO_KEYS",
          trial_duration: 200,
          post_trial_gap: 166, // SOA - trial_duration = 366ms - 200ms
        });
      });

      // After the RSVP sequence, present the question
      trials.push({
        type: jsPsychHtmlKeyboardResponse,
        stimulus: `<p style="font-size: 28px;">${prompt}</p>`,
        choices: "NO_KEYS",
        trial_duration: 200, // Show question for 200ms
      });

      // Then present the response options
      trials.push({
        type: jsPsychHtmlKeyboardResponse,
        stimulus: "<p>Press the left arrow for option 1 or the right arrow for option 2</p>",
        choices: ["ArrowLeft", "ArrowRight"],
        trial_duration: 5000, // Allow 500ms for response
        prompt: "<p>Option 1: [Details] &nbsp;&nbsp; Option 2: [Details]</p>", // Replace [Details] with actual options if available
        on_finish: function(data){
          data.choice = data.response;
        }
      });

      return trials;
    } else {
      return [];
    }
  }

  // Load the stimuli from the CSV file and create tasks
  loadStimuliFromCSV('../../stimuli/greek_sentences.csv').then((data) => {
    let task = [];
    data.forEach((row, index) => {
      if(row.length >= 2){
        const sentence = row[1];
        const prompt = row[2];
        const rsvpTrials = createRSVPTrials(sentence, prompt);
        task = task.concat(rsvpTrials);
      } else {
        console.warn(`Row ${index + 1} in CSV does not have enough columns.`);
      }
    });

    // Run the experiment
    jsPsych.run([instructions].concat(task));
  }).catch((error) => {
    console.log(error);
    console.error("Failed to start the experiment due to error in loading stimuli.");
  });
};
