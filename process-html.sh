#!/bin/bash
# filepath: process-html.sh

# Process each HTML file in the docs directory
for file in docs/*.html; do
    # Replace the HTML comment markers with actual content
    sed -i '' -e '/% BEGIN_HTML_ONLY/d' \
              -e '/% END_HTML_ONLY/d' \
              -e 's/% </</' \
              "$file"
    
    echo "Processed $file"
done