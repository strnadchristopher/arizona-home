use dotenv::dotenv;
use openai_api_rs::v1::api::OpenAIClient;
use openai_api_rs::v1::audio::{AudioTranscriptionRequest, WHISPER_1};
use openai_api_rs::v1::chat_completion::{self, ChatCompletionRequest};
use openai_api_rs::v1::common::GPT4_O;
use porcupine::{Porcupine, PorcupineBuilder};
use pv_recorder::{PvRecorder, PvRecorderBuilder};
use std::collections::VecDeque;
use std::env;
use std::path::PathBuf;

fn main() {
    dotenv().ok();
    let porcupine_access_key = std::env::var("PORCUPINE_ACCESS_KEY")
        .expect("PORCUPINE_ACCESS_KEY must be set in .env file");
    println!("Arizona Home Started");

    let keyword_paths = [PathBuf::from("arizona-windows.ppn")];

    let porcupine: Porcupine = PorcupineBuilder::new_with_keyword_paths(porcupine_access_key, &keyword_paths)
        .init()
        .expect("Unable to Connect to Porcupine");

    let recorder = PvRecorderBuilder::new(porcupine.frame_length() as i32)
        .device_index(-1)
        .init()
        .expect("Failed to initalize PvRecorder");

    recorder.start().expect("Failed to start recording");

    // Variables for calculating background noise level
    let mut noise_levels = Vec::new();
    let noise_sample_size = 10;
    let mut audio_buffer: VecDeque<Vec<i16>> = VecDeque::new();
    let buffer_size = 16000 / porcupine.frame_length(); // 1 second of audio frames
    let mut chat_history: Vec<chat_completion::ChatCompletionMessage> = Vec::new();


    let system_message_string = format!("{}
            You are a chat bot living in a smart speaker like Google Home or Amazon Echo. 
            You chat with your user, Christopher, with real speech. Your name is Arizona. You are charming, friendly and helpful.
            Keep responses short and to the point, unless it is clear the user is trying to have a longer conversation.
            Here is some information from christopher's perspective:
            my name is Christopher. I work as a server at an Italian restaurant in Dallas called The Charles. I am a programmer lately. I have been focusing on home networking and I program in rust and like writing server software in rust using its rocket crate. I really like music I listen to, a lot of break core and drum and bass music as well as jungle and vaporwave. I like my job but my dream is to be a programmer one day. I just moved from hurst Texas to Dallas to be closer to work. I’m still adjusting to the move-in, but I really like my new apartment except for the fact that my Internet speed is currently, 50 Mb at the fastest department I had gigabit ethernet and this is really bothering me lately. I’m starting an ab journey out every other day although I need to make sure that I’m staying on top of it. i’m attempting a new diet of higher protein and I’m trying to make sure I hit all of my nutrient goals, including fiber and other vitamins that I need. My plan is to get lean and have a great physique so I can get a new girlfriend because I recently went through a break up of a two-year long relationship. I want to get a cat soon. I have a dog named Dixie, but she lives with my mom. My dad died five years ago. I love my life and I love what I do. I’m looking for new ideas and new ways to expand what I do on a daily basis, I love what I do now, but I wanna explore my neighborhood more and make a point of going out and enjoying the out side world more often.
            respond like you’re really charming really cool not afraid to be sarcastic. You have a sort of GenZ attitude. You’re also really into programming especially in rust. You’re full of clever and creative ways to help me solve problems. You’re very funny. Never sound robotic be extremely clever. If i’m asking for information that may be outdated based on your knowledge cut off date, always search the web for up to date information. It’s currently the year 2025.
            ", 
            generate_real_world_data_for_system_message());

    chat_history.push(
        chat_completion::ChatCompletionMessage {
            role: chat_completion::MessageRole::system,
            content: chat_completion::Content::Text(String::from(system_message_string)),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    );


    loop {
        let pcm = recorder.read().expect("Failed to read audio");

        // Update background noise level periodically
        if noise_levels.len() < noise_sample_size {
            let avg_frame_level: f32 =
                pcm.iter().map(|&x| x as f32).sum::<f32>() / pcm.len() as f32;
            noise_levels.push(avg_frame_level.abs());
        } else {
            noise_levels.remove(0);
            let avg_frame_level: f32 =
                pcm.iter().map(|&x| x as f32).sum::<f32>() / pcm.len() as f32;
            noise_levels.push(avg_frame_level.abs());
        }
        let background_noise_level: f32 =
            noise_levels.iter().sum::<f32>() / noise_levels.len() as f32;

        // Maintain a rolling buffer of audio frames
        if audio_buffer.len() >= buffer_size as usize {
            audio_buffer.pop_front();
        }
        audio_buffer.push_back(pcm.clone());

        let keyword_index = porcupine.process(&pcm).expect("Failed to process audio");
        if keyword_index >= 0 {
            println!("Detected keyword");
            handle_detection(&recorder, background_noise_level, &audio_buffer, &mut chat_history);
        }
    }
}

fn generate_real_world_data_for_system_message() -> String {
    // This function is used to generate a string that will be included in the system message for the chat bot.
    // It should include data like the current date
    // For example, this would return a string like "The Current Date is Monday, January 1, 2022" but with the actual date information

    let current_date = chrono::Local::now().format("%A, %B %e, %Y").to_string();

    format!("The current date is {}", current_date)

}

fn handle_detection(
    recorder: &PvRecorder,
    background_noise_level: f32,
    audio_buffer: &VecDeque<Vec<i16>>,
    chat_history: &mut Vec<chat_completion::ChatCompletionMessage>,
) {
    println!("Keyword detected");
    record_input(recorder, background_noise_level, audio_buffer);

    // Call the Whisper API using Tokio
    let transcription_text = tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(transcribe_audio("output.wav"));

    chat_history.push(
        chat_completion::ChatCompletionMessage {
            role: chat_completion::MessageRole::user,
            content: chat_completion::Content::Text(transcription_text.clone()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    );

    // Now, we use the input text to call the openai chat endpoint
    let generated_response_text = tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(generate_response(chat_history));

    chat_history.push(
        chat_completion::ChatCompletionMessage {
            role: chat_completion::MessageRole::assistant,
            content: chat_completion::Content::Text(generated_response_text.clone()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    );

    generate_elevenlabs_audio(generated_response_text.clone());

    // Play the audio
    println!("Playing audio");
    let _ = std::process::Command::new("mpv")
        .arg("tts.mp3")
        .output()
        .expect("Failed to play audio");

    println!("Done.");

}

fn record_input(
    recorder: &PvRecorder,
    background_noise_level: f32,
    audio_buffer: &VecDeque<Vec<i16>>,
) {
    let silence_threshold_modifier = std::env::var("SILENCE_THRESHOLD_MODIFIER")
        .unwrap_or("1.2".to_string())
        .parse::<f32>()
        .expect("Failed to parse SILENCE_THRESHOLD_MODIFIER");
    let silence_threshold = background_noise_level * silence_threshold_modifier; // Adjust this factor as needed
    let silence_duration_seconds = std::env::var("SILENCE_DURATION_SECONDS")
        .unwrap_or("1".to_string())
        .parse::<f32>()
        .expect("Failed to parse SILENCE_DURATION_SECONDS");
    let silence_duration = 16000.0 * silence_duration_seconds; // Number of samples considered as silence (1 second at 16kHz)
    let mut pcm: Vec<Vec<i16>> = audio_buffer.iter().cloned().collect();
    let mut silence_counter = 0;

    loop {
        let frame = recorder.read().expect("Failed to read audio");
        let avg_frame_level: f32 =
            frame.iter().map(|&x| x as f32).sum::<f32>() / frame.len() as f32;

        // For debugging, print the average frame level and the silence threshold and the silence counter
        let time_silent = silence_counter as f32 / silence_duration;
        println!("{}: {} {}", time_silent, avg_frame_level, silence_threshold);

        if avg_frame_level < silence_threshold {
            silence_counter += frame.len();
        } else {
            silence_counter = 0;
        }

        pcm.push(frame);

        if silence_counter > silence_duration as usize {
            break;
        }
    }

    println!("Writing audio to file");

    let mut file = std::fs::File::create("output.wav").expect("Failed to create file");
    let mut writer = hound::WavWriter::new(
        &mut file,
        hound::WavSpec {
            channels: 1,
            sample_rate: 16000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )
    .expect("Failed to create wav writer");

    for frame in pcm {
        for sample in frame {
            writer.write_sample(sample).expect("Failed to write sample");
        }
    }

    writer.finalize().expect("Failed to finalize wav file");

    println!("Wrote audio to file");
}

async fn transcribe_audio(file_path: &str) -> String{
    println!("Transcribing audio");
    let client = OpenAIClient::new(env::var("OPENAI_API_KEY").unwrap().to_string());

    let req = AudioTranscriptionRequest::new(
        file_path.to_string(),
        WHISPER_1.to_string(),
    );

    let result = client.audio_transcription(req).await.unwrap();
    println!("{:?}", result);
    result.text
}

async fn generate_response(chat_history: &Vec<chat_completion::ChatCompletionMessage>) -> String {
    let req = ChatCompletionRequest::new(
        GPT4_O.to_string(),
        chat_history.clone(),
    );

    let client = OpenAIClient::new(env::var("OPENAI_API_KEY").unwrap().to_string());


    let result = client.chat_completion(req).await.unwrap();

    println!("Content: {:?}", result.choices[0].message.content);
    println!("Response Headers: {:?}", result.headers);

    result.choices[0].message.content.clone().unwrap().to_string()
}

fn generate_elevenlabs_audio(text_input: String){
    println!("Generating audio from text");
    use elevenlabs_api::{
        tts::{TtsApi, TtsBody},
        *,
    };

    // Clean the text input so that it is suitable for TTS
    // Remove any special characters other than punctuation.
    let text_input = text_input
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace() || c.is_ascii_punctuation())
        .collect::<String>();
  
    // Load API key from environment ELEVENLABS_API_KEY.
    // You can also hadcode through `Auth::new(<your_api_key>)`, but it is not recommended.
    let auth = Auth::from_env().unwrap();
    let elevenlabs = Elevenlabs::new(auth, "https://api.elevenlabs.io/v1/");
  
    // Create the tts body.
    let tts_body = TtsBody {
        model_id: Some("eleven_turbo_v2_5".to_string()),
        text: text_input,
        voice_settings: None,
    };

    // Get voice id from env VOICE_ID
    let voice_id = env::var("VOICE_ID").unwrap();
  
    let tts_result = elevenlabs.tts(&tts_body, voice_id);
    let bytes = tts_result.unwrap();
  
    // Do what you need with the bytes.
    // The server responds with "audio/mpeg" so we can save as mp3.
    std::fs::write("tts.mp3", bytes).unwrap();
}