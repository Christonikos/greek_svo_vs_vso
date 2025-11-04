window.onload = function() {
  console.log("Page loaded. Initializing experiment...");

  const jsPsych = initJsPsych({
    on_finish: function() {
      // After the experiment ends, save the data to a new CSV file
      saveDataToCSV();
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

  // Function to shuffle an array (Fisher-Yates algorithm)
  function shuffleArray(array) {
    for (let i = array.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [array[i], array[j]] = [array[j], array[i]];
    }
    return array;
  }

  // Function to create RSVP trials from a sentence with randomized response options
  function createRSVPTrials(sentence, prompt, correctResponse, falseResponse, rowIndex) {
    let trials = [];
    const words = sentence.trim().split(/\s+/); // Split by any whitespace

    if ((sentence != "") & (sentence != "Sentence")) {
      console.log(`Creating RSVP trials for sentence: "${sentence}" with prompt: "${prompt}"`);

      // Display each word for 200ms, with a SOA of 366ms
      words.forEach((word) => {
        trials.push({
          type: jsPsychHtmlKeyboardResponse,
          stimulus: `<p style="font-size: 32px;">${word}</p>`,
          choices: "NO_KEYS",
          trial_duration: 200,
          post_trial_gap: 166, // SOA - trial_duration = 366ms - 200ms
        });
      });

      // Randomize response positions
      let firstOption, secondOption;
      if (Math.random() < 0.5) {
        firstOption = correctResponse;
        secondOption = falseResponse;
      } else {
        firstOption = falseResponse;
        secondOption = correctResponse;
      }

      // Present the actual prompt and response options
      trials.push({
        type: jsPsychHtmlKeyboardResponse,
        stimulus: `<p style="font-size: 36px;">${prompt}</p>`,  // Increase font size for question
        choices: ["ArrowLeft", "ArrowRight"],
        trial_duration: 5000, // Allow 5000ms for response
        prompt: `<p style="font-size: 32px;">${firstOption} &nbsp;&nbsp;&nbsp;&nbsp; ${secondOption}</p>`,  // Increase font size for options
        on_finish: function(data){
          if (data.response === null) {
            data.correct = false;  // No response is treated as incorrect
            data.reaction_time = null;  // No reaction time if no response
          } else {
            data.correct = (data.response === "ArrowLeft" && firstOption === correctResponse) ||
                           (data.response === "ArrowRight" && secondOption === correctResponse);
            data.reaction_time = data.rt; // Store the reaction time
          }

          // Store data in global variable for later CSV export
          csvData[rowIndex].push(data.reaction_time, data.correct ? "correct" : "wrong");
        }
      });

      // Add fixation cross based on correctness
      trials.push({
        type: jsPsychHtmlKeyboardResponse,
        stimulus: function() {
          if (jsPsych.data.getLastTrialData().values()[0].correct) {
            return '<p style="font-size: 48px; color: green;">+</p>'; // Green cross for correct
          } else {
            return '<p style="font-size: 48px; color: red;">+</p>'; // Red cross for incorrect
          }
        },
        choices: "NO_KEYS",
        trial_duration: 1000, // Display the cross for 1000ms
      });

      return trials;
    } else {
      return [];
    }
  }

  // Function to save the data to a CSV file
  function saveDataToCSV() {
    const csvContent = "data:text/csv;charset=utf-8," + 
      csvData.map(e => e.join(",")).join("\n");
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", "experiment_results.csv");
    document.body.appendChild(link);
    link.click();
  }

  let csvData = [];

  // Load the stimuli from the CSV file and create tasks
  loadStimuliFromCSV('../../stimuli/greek_sentences.csv').then((data) => {
    let task = [];
    csvData = data.slice(); // Make a copy of the original data
    data.forEach((row, index) => {
      if(row.length >= 10){
        const sentence = row[1];
        const prompt = row[2];
        const correctResponse = row[8];
        const falseResponse = row[9];
        const rsvpTrials = createRSVPTrials(sentence, prompt, correctResponse, falseResponse, index);
        task.push(rsvpTrials);
      } else {
        console.warn(`Row ${index + 1} in CSV does not have enough columns.`);
      }
    });

    // Randomize the order of the trials (i.e., sentences)
    task = shuffleArray(task);

    // Flatten the array since each trial returns an array of tasks
    task = task.flat();

    // Run the experiment
    jsPsych.run([instructions].concat(task));
  }).catch((error) => {
    console.log(error);
    console.error("Failed to start the experiment due to error in loading stimuli.");
  });
};
