import { useId } from "react";
import type { TextareaHTMLAttributes } from "react";

interface TextAreaProps extends TextareaHTMLAttributes<HTMLTextAreaElement> {
  label: string;
  hint?: string;
}

export function TextArea({ label, hint, id, className = "", ...props }: TextAreaProps) {
  const generatedId = useId();
  const textareaId = id ?? generatedId;
  return (
    <label className="flex flex-col gap-1 text-sm text-neutral-300" htmlFor={textareaId}>
      <span className="font-medium">{label}</span>
      <textarea
        id={textareaId}
        className={`min-h-24 rounded border border-neutral-700 bg-neutral-900 px-3 py-2 text-neutral-100 focus:border-sky-500 focus:outline-none ${className}`}
        {...props}
      />
      {hint ? <span className="text-xs text-neutral-500">{hint}</span> : null}
    </label>
  );
}
