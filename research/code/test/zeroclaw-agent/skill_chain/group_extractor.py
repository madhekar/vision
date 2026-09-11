import time
from selenium import webdriver
from selenium.webdriver.firefox.service import Service as FirefoxService
from webdriver_manager.firefox import GeckoDriverManager

def extract_whatsapp_group_ids():
    # 1. Initialize the Firefox driver
    print("Launching Firefox...")
    options = webdriver.FirefoxOptions()
    # Optional: Use a persistent profile to avoid scanning the QR code every time
    # options.add_argument("-profile")
    # options.add_argument("/path/to/your/firefox/profile")
    
    driver = webdriver.Firefox(service=FirefoxService(GeckoDriverManager().install()), options=options)
    
    # 2. Open WhatsApp Web
    driver.get("https://whatsapp.com")
    print("Please scan the QR code to log in.")
    
    # 3. Wait for the user to login and the page to fully load
    # Adjust the sleep time if your chats take longer to load
    time.sleep(25) 
    
    # 4. Execute JavaScript to extract group data from WhatsApp's internal store
    js_script = """
    const chats = window.Store ? window.Store.Chat.models : [];
    if (chats.length === 0) {
        // Fallback if Store isn't directly exposed on window
        try {
            const require = window.require || window.webpackChunkwhatsapp_web_client;
            if (require) {
                // Attempt to find the chat store dynamically
                const chatModule = window.Debug?.models?.Chat || 
                                   Object.values(window).find(x => x && x.Chat)?.Chat;
                if (chatModule && chatModule.models) {
                    return chatModule.models
                        .filter(chat => chat.isGroup)
                        .map(g => ({ id: g.id._serialized, name: g.name }));
                }
            }
        } catch (e) {
            return { error: e.message };
        }
    }
    
    // Standard extraction if window.Store is accessible
    return chats
        .filter(chat => chat.isGroup)
        .map(g => ({
            id: g.id._serialized || g.id,
            name: g.formattedTitle || g.name
        }));
    """
    
    print("Extracting group IDs...")
    groups = driver.execute_script(js_script)
    
    # 5. Output the results
    if not groups:
        print("No groups found. Ensure you are fully logged in and chats have loaded.")
    elif isinstance(groups, dict) and "error" in groups:
        print(f"Error executing script: {groups['error']}")
    else:
        print(f"\nFound {len(groups)} groups:\n")
        print(f"{'Group Name':<40} | {'Group ID'}")
        print("-" * 80)
        for group in groups:
            print(f"{str(group['name']):<40} | {group['id']}")
            
    # Keep browser open for a moment before closing
    time.sleep(5)
    driver.quit()

if __name__ == "__main__":
    extract_whatsapp_group_ids()
