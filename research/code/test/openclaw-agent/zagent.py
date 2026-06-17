'''
Architecture
┌─────────────────────────────────────────┐
│         VSCode Extension                │
│  ┌────────────┐      ┌──────────────┐   │
│  │ Chat UI    │      │ Context      │   │
│  │ (Webview)  │      │ Tracker      │   │
│  └─────┬──────┘      └──────┬───────┘   │
│        │                    │           │
│        └────────┬───────────┘           │
│                 │ WebSocket             │
└─────────────────┼───────────────────────┘
                  │
         ┌────────▼────────┐
         │  Gateway :18789 │  ← OpenClaw Core
         │  (TypeScript)   │
         └────────┬────────┘
                  │
    ┌─────────────┼─────────────┐
    │             │             │
┌───▼───┐    ┌────▼────┐    ┌───▼────┐
│ Agent │    │ Skills  │    │ Memory │
│Runner │    │ (Tools) │    │ Store  │
└───────┘    └─────────┘    └────────┘


[6/15/26 2:24 PM] BotFather: I can help you create and manage Telegram bots. If you're new to the Bot API, please see the manual (https://core.telegram.org/bots).

You can control me by sending these commands:

/newbot - create a new bot
/mybots - edit your bots

Edit Bots
/setname - change a bot's name
/setdescription - change bot description
/setabouttext - change bot about info
/setuserpic - change bot profile photo
/setcommands - change the list of commands
/deletebot - delete a bot

Bot Settings
/token - get authorization token
/revoke - revoke bot access token
/setinline - toggle inline mode (https://core.telegram.org/bots/inline)
/setinlinegeo - toggle inline location requests (https://core.telegram.org/bots/inline#location-based-results)
/setinlinefeedback - change inline feedback (https://core.telegram.org/bots/inline#collecting-feedback) settings
/setjoingroups - can your bot be added to groups?
/setprivacy - toggle privacy mode (https://core.telegram.org/bots/features#privacy-mode) in groups

Web Apps
/myapps - edit your web apps (https://core.telegram.org/bots/webapps)
/newapp - create a new web app (https://core.telegram.org/bots/webapps)
/listapps - get a list of your web apps
/editapp - edit a web app
/deleteapp - delete an existing web app

Games
/mygames - edit your games (https://core.telegram.org/bots/games)
/newgame - create a new game (https://core.telegram.org/bots/games)
/listgames - get a list of your games
/editgame - edit a game
/deletegame - delete an existing game
[6/15/26 2:25 PM] Bo Mad: /newbot
[6/15/26 2:25 PM] BotFather: Alright, a new bot. How are we going to call it? Please choose a name for your bot.
[6/15/26 2:25 PM] Bo Mad: zmedia
[6/15/26 2:25 PM] BotFather: Good. Now let's choose a username for your bot. It must end in `bot`. Like this, for example: TetrisBot or tetris_bot.
[6/15/26 2:26 PM] Bo Mad: zm_bot
[6/15/26 2:26 PM] BotFather: Sorry, this username is already taken. Please try something different.
[6/15/26 2:26 PM] Bo Mad: zmedia_bot
[6/15/26 2:26 PM] BotFather: Done! Congratulations on your new bot. You will find it at t.me/zmedia_bot. You can now add a description, about section and profile picture for your bot, see /help for a list of commands. By the way, when you've finished creating your cool bot, ping our Bot Support if you want a better username for it. Just make sure the bot is fully operational before you do this.

Use this token to access the HTTP API:
8823078829:AAGmg-lIQ6ZrOUwPqTn0aN9nlfBG00a9Nnw
Keep your token secure and store it safely, it can be used by anyone to control your bot.

For a description of the Bot API, see this page: https://core.telegram.org/bots/api

'''

'''
In ZeroClaw, an agent's identity and behavioral traits are defined using standardized, portable files in the agent's workspace directory. 
Instead of hardcoding instructions into system prompts, 
ZeroClaw uses a set of specific markdown and JSON configuration files to shape how your agent acts and operates.
The identity and personality configuration for a ZeroClaw agent is governed by the following core text files:IDENTITY.md: Defines the agent's primary identity, name, role, and overarching purpose.SOUL.md: Sets the agent's core personality, tone, ethics, and fundamental values.USER.md: Describes the human user, 
including your preferences, background, and what the agent is actively helping you with.AGENTS.md: Contains specific behavior guidelines, operating boundaries,
 and execution instructions.Identity Systems: OpenClaw vs. AIEOSZeroClaw provides two approaches for structuring agent identity:OpenClaw 
 (Default): Uses traditional markdown files in your workspace (IDENTITY.md, SOUL.md, etc.). If you are migrating from OpenClaw,
   you can transfer these settings natively using the zeroclaw migrate openclaw command.AIEOS (AI Entity Object Specification): 
   A standardized JSON format. By configuring your config.toml, you can set your agent to read an identity.json file. AIEOS enables 
   you to export and import your agent's exact persona and behavior across multiple AIEOS-compatible systems without losing its traits.
   Multi-Agent ContextIf you are running multiple agents on a single gateway, ZeroClaw treats each agent alias as a completely separate entity. 
   Each gets its own identity files, workspace directories, risk profile, and memory scope.If you would like to customize these settings, 
   I can help you:Format the IDENTITY.md or SOUL.md for a specific personality type (e.g., professional, concise, or creative).Set up an AIEOS JSON 
   payload.Configure AGENTS.md to establish boundaries and guidelines.Let me know which of these you'd like to dive into!

'''

''''
San Diego, California, is world-renowned for its exceptionally mild and pleasant climate. It primarily features a **Mediterranean climate** 
(specifically the *Csb* subtype under the Köppen classification), characterized by warm, dry summers and mild, relatively wet winters.

The city’s weather is heavily influenced by its proximity to the Pacific Ocean, which acts as a massive temperature regulator. 

Here is a detailed breakdown of the various components of San Diego's climate:

### 1. Temperature Patterns
San Diego is known for having very little seasonal temperature extreme compared to other parts of the United States.
*   **Mild Winters:** Even in the coldest months (December through February), daytime temperatures rarely drop below the 60s°F (15–20°C). 
Frost is extremely rare near the coast and virtually non-existent in the city proper.
*   **Warm to Hot Summers:** During the summer (July through September), temperatures are generally warm but rarely reach the scorching heat 
seen in inland Southern California. Coastal areas typically stay in the 70s or low 80s°F (21–28°C). 
*   **Coastal vs. Inland Gradient:** There is a significant "microclimate" difference between the coast and the inland valleys (like Escondido 
or El Cajon). While the coast remains cool, inland areas can experience much higher temperatures, sometimes exceeding 90–100°F (32–38°C) during heatwaves.

### 2. The Marine Layer (Coastal Influence)
Perhaps the most defining feature of San Diego's climate is the **marine layer**.
*   **Fog and Low Clouds:** During late spring and summer, moisture from the Pacific Ocean often creates a thick layer of low clouds or fog that moves inland. 
This provides a natural "air conditioning" effect for coastal neighborhoods.
*   **The "June Gloom":** San Diego experiences a phenomenon known as "June Gloom," where overcast skies persist throughout much of the morning and
 sometimes the entire day during the month of June. 
*   **Coastal Breezes:** Daytime onshore winds (blowing from sea to land) help keep coastal temperatures stable, while nighttime offshore breezes 
can allow for clearer, warmer nights.

### 3. Precipitation and Humidity
San Diego is a relatively arid region, which contributes to its "dry" feel compared to tropical or even Eastern US climates.
*   **Seasonal Rainfall:** Most precipitation occurs during the winter months (December through March). Summers are almost entirely rainless.
*   **Low Annual Totals:** The city receives an average of only about 10 inches (25 cm) of rain per year.
*   **Humidity Levels:** Because of the ocean, humidity is moderate. While the summer can feel slightly more humid due to the marine layer, 
it lacks the oppressive, heavy humidity found in places like Florida or the American South.

### 4. Summary Table of Seasonal Characteristics

| Feature | Winter (Dec–Feb) | Spring (Mar–May) | Summer (Jun–Aug) | Fall (Sep–Nov) |
| :--- | :--- | :--- | :--- | :--- |
| **Temperature** | Mild/Cool | Transitioning to Warm | Warm/Hot | Warm/Mild |
| **Sky Conditions** | Often overcast/Rainy | Increasing sunshine | "June Gloom" / High Sun | Clear/Sunny |
| **Precipitation** | Highest frequency | Moderate | Very Low | Very Low |
| **Sea Breeze** | Gentle | Moderate | Strongest (Marine Layer) | Variable |

In short, San Diego's climate is defined by **stability**. The lack of extreme cold and the avoidance of extreme humidity make it one of the most 
sought-after locations for people seeking year-round comfortable living.


'''

'''
cat <<EOF > ModelFile
FROM gemma4:26b
PARAMETER num_ctx 131072
PARAMETER num_gpu 999
EOF
'''



'''
Configuring ChromaDB for ZeroClaw skills involves spinning up a ChromaDB container, configuring an embedding model, and updating your config.toml file to integrate it with the chromadb-memory skill.Step 1: Run ChromaDB and Embedding ModelsZeroClaw relies on an external vector database for its advanced memory and skill retrieval capabilities.Start ChromaDB (via Docker): Run the following command in your terminal:bashdocker run -d --name chromadb -p 8100:8000 chromadb/chroma:latest
Pull an Embedding Model: ZeroClaw requires a text embedder. Pull the nomic-embed-text model via Ollama with:bashollama pull nomic-embed-text
Step 2: Configure config.tomlOpen your ZeroClaw configuration file located at ~/.zeroclaw/config.toml and define the ChromaDB memory connection parameters:toml[skills.chromadb]
chromaUrl = "http://localhost:8100"
ollamaUrl = "http://localhost:11434"
embeddingModel = "nomic-embed-text"

# Define the UUID for your specific ChromaDB collection
collectionId = "your-collection-uuid-here" 
Step 3: Enable Open Skills and Skill InvocationZeroClaw requires explicit permission to synchronize and load community or external skills.Enable open skills in your config.toml:toml[skills]
open_skills_enabled = true
prompt_injection_mode = "compact"
Once configured, you can download specific database skills, such as chromadb-memory, from the LobeHub Skills Marketplace or the Claude MCP Directory.Install the skill using the ZeroClaw CLI:bashzeroclaw skills install chromadb-memory

'''

'''
nomic-embed-text is a state-of-the-art, open-source text embedding model developed by Nomic AI. 
It excels at capturing the semantic meaning of text into numerical vectors and is widely used for Retrieval-Augmented Generation (RAG), 
semantic search, and text clustering.Key FeaturesMassive Context Window: Supports an 8,192-token context length, making it ideal for processing 
large document chunks.Open Source & Reproducible: The weights, training code, and curated training data are fully open and auditable.
High Performance: Consistently outperforms closed-source models like OpenAI text-embedding-ada-002 and text-embedding-3-small on major short- and 
long-context benchmarks.Multilingual Mixture-of-Experts: Newer versions (like v2-MoE) provide robust performance across roughly 100 languages.How 
to Use ItBecause of its open nature, you can deploy and run nomic-embed-text in multiple ways depending on your architecture:Locally via Ollama: 
You can easily run it on your own machine. First, install Ollama, then run ollama run nomic-embed-text in your terminal to generate embeddings.Via 
Nomic API: You can use the Nomic Atlas Embedding API, which allows you to process production workloads with 1 million free tokens included.
Via Hugging Face: You can download the weights and view model documentation directly on the Nomic Hugging Face Hub.Coding ExampleWhen using it 
programmatically with the official Nomic Python client or Hugging Face, you must specify a prefix (task type) to get the best results. Common task 
prefixes include:search_document: For embedding document chunks prior to search.search_query: For embedding user search terms.classification or clustering: 
For analytical data processing.For more implementation options and documentation, check out the Nomic Text Embeddings Documentation.
'''