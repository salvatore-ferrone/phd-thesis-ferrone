io.stderr:write("Code filter loaded\n")

function RawBlock(el)
  io.stderr:write("RawBlock received:\n" .. el.text .. "\n")

  -- Match lstlisting with language
  local lang, code = el.text:match("\\begin{lstlisting}%s*%[%s*language%s*=%s*(%w+)%s*%]%s*\n?(.-)\\end{lstlisting}")
  if lang and code then
    return pandoc.CodeBlock(code, {["class"] = "language-" .. string.lower(lang)})
  end

  -- Match lstlisting without language
  code = el.text:match("\\begin{lstlisting}%s*\n?(.-)\\end{lstlisting)")
  if code then
    return pandoc.CodeBlock(code)
  end

  -- Match verbatim
  code = el.text:match("\\begin{verbatim}%s*(.-)\\end{verbatim)")
  if code then
    return pandoc.CodeBlock(code)
  end

  -- If it's raw HTML (like your <video> block), return as HTML
  return pandoc.RawBlock("html", el.text)
end
