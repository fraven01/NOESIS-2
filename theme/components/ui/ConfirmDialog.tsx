import * as React from "react";
import { Dialog } from "./Dialog";
import { Button } from "./Button";
import { Input } from "./Input";
import { cn } from "./cn";

/**
 * Dialog for confirming destructive actions. Optionally require typing text to confirm.
 * @example
 * <ConfirmDialog open={open} onClose={...} onConfirm={...} confirmText="DELETE" />
 */
export interface ConfirmDialogProps extends React.HTMLAttributes<HTMLDivElement> {
  open: boolean;
  onClose: () => void;
  onConfirm: () => void;
  title: string;
  description?: string;
  confirmLabel?: string;
  cancelLabel?: string;
  confirmText?: string;
}

export const ConfirmDialog: React.FC<ConfirmDialogProps> = ({
  open,
  onClose,
  onConfirm,
  title,
  description,
  confirmLabel = "Delete",
  cancelLabel = "Cancel",
  confirmText,
  className,
  ...props
}) => {
  const [value, setValue] = React.useState("");

  React.useEffect(() => {
    if (open) {
      const previous = document.body.style.overflow;
      document.body.style.overflow = "hidden";
      return () => {
        document.body.style.overflow = previous;
      };
    }
  }, [open]);

  React.useEffect(() => {
    if (!open) {
      setValue("");
    }
  }, [open]);

  React.useEffect(() => {
    setValue("");
  }, [confirmText]);

  const disabled = confirmText ? value !== confirmText : false;

  return (
    <Dialog open={open} onClose={onClose} ariaLabel={title}>
      <div className={cn("space-y-4", className)} {...props}>
        <h2 className="text-lg font-semibold">{title}</h2>
        {description && <p>{description}</p>}
        {confirmText && (
          <Input
            aria-label="Type to confirm"
            placeholder={confirmText}
            value={value}
            onChange={(e) => setValue(e.target.value)}
          />
        )}
        <div className="flex justify-end gap-2 pt-2">
          <Button variant="secondary" onClick={onClose}>
            {cancelLabel}
          </Button>
          <Button
            variant="destructive"
            disabled={disabled}
            onClick={() => {
              onConfirm();
              onClose();
            }}
          >
            {confirmLabel}
          </Button>
        </div>
      </div>
    </Dialog>
  );
};
