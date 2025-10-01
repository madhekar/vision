import re

escaped = re.escape("I'm going to NJ.")                    
print(escaped)     

escaped = "I'm going".translate(str.maketrans({
                                          "-":  r"\-",
                                          "]":  r"\]",
                                          "\\": r"\\",
                                          "^":  r"\^",
                                          "$":  r"\$",
                                          "*":  r"\*",
                                          ".":  r"\.",
                                          "'":  r"\'",
                                          '"':  r"\""
                                          }))

print(escaped) 