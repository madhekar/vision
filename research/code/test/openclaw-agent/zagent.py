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