#this file updates the readme to include the current date/time
from dateutil.tz import tzlocal
from datetime import datetime


with open('README.template.md') as f:
    contents = f.read()
    
today = datetime.now(tz=tzlocal())
datestr = today.strftime('%B %e %Y  %I:%M %p %Z')
contents = contents.replace('{%Date Here%}', datestr)

with open('README.md', 'w') as f:
    f.write(contents)
    
    
print("Updated README.md with the date: ", datestr)
