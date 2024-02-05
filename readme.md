# ANTicle
A little tool that carrys a massive weight


AI-based tool for the generation of news articles based on news agency reports, social media posts and other sources.
Feed the tool with information and let it generate everything you need to publish a news article.
Currently, in development with a working backend. You are welcome to download and try it! 
An easy to use frontend is in the making.


## 📘 License

ANTicle is licensed under the MIT license. For more information check the LICENSE file for details.

## Features

### Working

- recognise text genre
- generate article text, titel, tagline, tester, SEO Tiel, SEO Teaser
- history of generated texts & amount of tokens used
- checking for AI hallucinations
- adaptive openAI API key update to avoid hitting errors
  
### Work in progress

- adding variables to the prompt
- prompts

### Planned

- Web based frontend

## Running it

  1. Choose your preferred IDE
  1. Install Python 3.11.6 or higher
  2. Clone the repository: 
     1. `git clone [https://github.com/ANTicle/main.git]`
  3. Add your OpenAI API key in the settings file 
     (you can add up to 3 from different accounts for going past the daily token limit of your account, but only 1 is needed)
     1. Find it here `[https://platform.openai.com/api-keys]`
  4. Add your input as *.txt, *.doc or *.docx to the Input_data folder
  5. Run main.py
  6. On the first run and after all updates, the script will start to be downloading and updating everything it needs
  7. Your Output can be found in the Output_data folder

Please keep in mind that this is still in development and currently focused on the backend. As of today only pompts for writing rudimentary weater reports are provided.
More and improved prompts for testing will follow in the comming weeks.


## 🙏 Supported by

- Media Tech Lab [`media-tech-lab`](https://github.com/media-tech-lab)

<a href="https://www.media-lab.de/en/programs/media-tech-lab">
    <img src="https://raw.githubusercontent.com/media-tech-lab/.github/main/assets/mtl-powered-by.png" width="240" title="Media Tech Lab powered by logo">
</a>

