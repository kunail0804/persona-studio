import Markdown, { type Components } from "react-markdown";

// The narrator writes Markdown; these classes keep the rendered elements
// reading as prose inside a chat bubble rather than as a document. Fenced
// blocks arrive as <pre><code class="language-…">, inline code as a bare
// <code> — the className is what tells them apart.
const components: Components = {
  h1: ({ children }) => <h1 className="mt-4 text-xl font-semibold">{children}</h1>,
  h2: ({ children }) => <h2 className="mt-4 text-lg font-semibold">{children}</h2>,
  h3: ({ children }) => <h3 className="mt-3 font-semibold">{children}</h3>,
  p: ({ children }) => <p className="my-2 leading-relaxed">{children}</p>,
  ul: ({ children }) => <ul className="my-2 list-disc pl-6">{children}</ul>,
  ol: ({ children }) => <ol className="my-2 list-decimal pl-6">{children}</ol>,
  li: ({ children }) => <li className="my-1">{children}</li>,
  blockquote: ({ children }) => (
    <blockquote className="my-2 border-l-2 border-neutral-600 pl-3 text-neutral-400 italic">
      {children}
    </blockquote>
  ),
  pre: ({ children }) => (
    <pre className="my-2 overflow-x-auto rounded border border-neutral-800 bg-neutral-950 p-3 text-sm">
      {children}
    </pre>
  ),
  code: ({ className, children }) =>
    className ? (
      <code className={className}>{children}</code>
    ) : (
      <code className="rounded bg-neutral-800 px-1 py-0.5 text-[0.9em]">{children}</code>
    ),
  a: ({ children, href }) => (
    <a href={href} className="text-sky-400 underline underline-offset-2" rel="noreferrer">
      {children}
    </a>
  ),
  hr: () => <hr className="my-3 border-neutral-800" />,
};

/**
 * Narration rendered as Markdown, always — from the first streamed fragment
 * to the last persisted message. There is no raw-text fallback while
 * streaming: the previous version had one and the bubble visibly beat
 * flipping between raw source and formatted text (see issue #10).
 *
 * react-markdown escapes HTML in the source by default, which is the issue's
 * second criterion — no `rehype-raw`, and no `dangerouslySetInnerHTML`
 * anywhere.
 */
export function Narration({ text }: { text: string }) {
  return (
    <div className="mt-2 text-neutral-100 [&>*:first-child]:mt-0 [&>*:last-child]:mb-0">
      <Markdown components={components}>{text}</Markdown>
    </div>
  );
}
