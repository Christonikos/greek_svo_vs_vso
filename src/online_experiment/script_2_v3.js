// Debugging flag
const DEBUG_MODE = true;
const DEBUG_TRIALS = 5;

let sessionCounter = 0;

function generateSubjectID() {
  sessionCounter++;
  const timestamp = new Date().toISOString().replace(/[-:]/g, '').replace('T', '_').split('.')[0];
  return `subject_${timestamp}_${sessionCounter}`;
}

function ensureResultsDirectory() {
  return new Promise((resolve, reject) => {
    const dirName = 'results';
    if (typeof window.showDirectoryPicker === 'function') {
      // For browsers that support the File System Access API
      window.showDirectoryPicker().then(handle => {
        handle.getDirectoryHandle(dirName, { create: true }).then(() => {
          resolve(dirName);
        }).catch(reject);
      }).catch(reject);
    } else {
      // For browsers that don't support the File System Access API
      // We can't create directories client-side, so we'll just resolve with the directory name
      resolve(dirName);
    }
  });
}

// Generate subject ID when the experiment starts
const subjectID = generateSubjectID();

window.onload = function() {
  console.log("Page loaded. Initializing experiment...");

  const jsPsych = initJsPsych({
    on_finish: function() {
      // Display completion message and save button
      document.body.innerHTML = `
        <h1>Experiment Complete</h1>
        <p>Thank you for participating. Click the button below to save your data.</p>
        <button id="saveButton">Save Data</button>
      `;
      document.getElementById('saveButton').addEventListener('click', saveDataToCSV);
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
        trial_duration: 1000, // Allow 1000ms for response
        prompt: `<p style="font-size: 32px;">${firstOption} &nbsp;&nbsp;&nbsp;&nbsp; ${secondOption}</p>`,  // Increase font size for options
        on_finish: function(data){
          let performance;
          if (data.response === null) {
            data.correct = false;
            data.reaction_time = null;
            performance = "did_not_answer";
          } else {
            data.correct = (data.response === "ArrowLeft" && firstOption === correctResponse) ||
                           (data.response === "ArrowRight" && secondOption === correctResponse);
            data.reaction_time = data.rt;
            performance = data.correct ? "correct" : "wrong";
          }

          // Store data in global variable for later CSV export
          csvData[rowIndex].push(subjectID, performance, data.reaction_time);
        }
      });

      // Add fixation cross based on correctness
      trials.push({
        type: jsPsychHtmlKeyboardResponse,
        stimulus: function() {
          const lastTrial = jsPsych.data.getLastTrialData().values()[0];
          if (lastTrial.response === null) {
            return '<p style="font-size: 48px; color: blue;">+</p>'; // Blue cross for no response
          } else if (lastTrial.correct) {
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
    ensureResultsDirectory().then(dirName => {
      // Add headers for new columns
      const headers = csvData[0].concat(["subject", "performance", "reaction_time"]);
      
      const csvContent = headers.join(",") + "\n" + 
        csvData.slice(1).map(e => e.join(",")).join("\n");
      
      const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
      const fileName = `${dirName}/${subjectID}.csv`;

      if (window.showSaveFilePicker) {
        window.showSaveFilePicker({
          suggestedName: fileName,
          types: [{
            description: 'CSV File',
            accept: { 'text/csv': ['.csv'] },
          }],
        }).then(fileHandle => {
          fileHandle.createWritable().then(writable => {
            writable.write(blob).then(() => writable.close());
          });
        }).catch(error => {
          console.error("Error saving file:", error);
          // Fallback to download if showSaveFilePicker fails
          downloadCSV(blob, fileName);
        });
      } else {
        // Fallback for browsers that don't support the File System Access API
        downloadCSV(blob, fileName);
      }
    }).catch(error => {
      console.error("Error saving results:", error);
      // Fallback to download if ensureResultsDirectory fails
      downloadCSV(new Blob([csvContent], { type: 'text/csv;charset=utf-8;' }), `${subjectID}.csv`);
    });
  }

  function downloadCSV(blob, fileName) {
    const link = document.createElement("a");
    if (link.download !== undefined) {
      const url = URL.createObjectURL(blob);
      link.setAttribute("href", url);
      link.setAttribute("download", fileName);
      link.style.visibility = 'hidden';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  }

  let csvData = [];

  // Load the stimuli from the CSV file and create tasks
  loadStimuliFromCSV('stimuli/greek_sentences.csv').then((data) => {
    let allTrials = [];
    csvData = data.slice(); // Make a copy of the original data
    
    // Shuffle the csvData array to randomize sentence order
    shuffleArray(csvData);
    
    // Limit the number of trials if in debug mode
    const trialsToRun = DEBUG_MODE ? Math.min(DEBUG_TRIALS, csvData.length) : csvData.length;
    
    for (let index = 0; index < trialsToRun; index++) {
      const row = csvData[index];
      if(row.length >= 10){
        const sentence = row[1];
        const prompt = row[2];
        const correctResponse = row[8];
        const falseResponse = row[9];
        const rsvpTrials = createRSVPTrials(sentence, prompt, correctResponse, falseResponse, index);
        allTrials = allTrials.concat(rsvpTrials);
      } else {
        console.warn(`Row ${index + 1} in CSV does not have enough columns.`);
      }
    }

    console.log("Experiment started");
    // Run the experiment
    jsPsych.run([instructions].concat(allTrials));
  }).catch((error) => {
    console.log(error);
    console.error("Failed to start the experiment due to error in loading stimuli.");
  });
};