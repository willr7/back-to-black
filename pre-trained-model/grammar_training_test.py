from language_tool_python import LanguageTool

tool = LanguageTool("en-US")

s = "Y'all ain't gon' believe this."

matches = tool.check(s)

print(len(matches))
print(matches)
print(matches[0])
print(matches[1])
print(matches[0])
