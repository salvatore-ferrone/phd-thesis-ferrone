function RawBlock(el)
    if el.format == "latex" then
      local filename = el.text:match("VIDEO:%s*(.+)")
      if filename and FORMAT:match("html") then
        return pandoc.RawBlock("html", string.format([[
  <div class="video-container">
    <video controls style="max-width: 100%%;">
      <source src="videos/%s" type="video/mp4">
      Your browser does not support the video tag.
    </video>
  </div>]], filename))
      else
        return {} -- Omit from PDF
      end
    end
  end
  
  function CodeBlock(el)
    local filename = el.text:match("VIDEO:%s*(.+)")
    if filename and FORMAT:match("html") then
      return pandoc.RawBlock("html", string.format([[
  <div class="video-container">
    <video controls style="max-width: 100%%;">
      <source src="videos/%s" type="video/mp4">
      Your browser does not support the video tag.
    </video>
  </div>]], filename))
    else
      return {} -- Omit from PDF
    end
  end