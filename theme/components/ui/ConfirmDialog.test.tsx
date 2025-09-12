import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { axe } from "jest-axe";
import { ConfirmDialog } from "./ConfirmDialog";
import { Button } from "./Button";
import { vi } from "vitest";

describe("ConfirmDialog", () => {
  it("requires typing to confirm", async () => {
    const onConfirm = vi.fn();
    const onClose = vi.fn();
    const { container } = render(
      <ConfirmDialog
        open
        onClose={onClose}
        onConfirm={onConfirm}
        title="Delete"
        confirmText="DELETE"
      />
    );
    const confirm = screen.getByRole("button", { name: "Delete" });
    expect(confirm).toBeDisabled();
    await userEvent.type(screen.getByLabelText(/type to confirm/i), "DELETE");
    expect(confirm).toBeEnabled();
    await userEvent.click(confirm);
    expect(onConfirm).toHaveBeenCalled();
    expect(await axe(container)).toHaveNoViolations();
  });

  it("traps focus and closes with Escape", async () => {
    const onClose = vi.fn();
    render(
      <ConfirmDialog
        open
        onClose={onClose}
        onConfirm={() => {}}
        title="Delete"
        confirmText="CONFIRM"
      />
    );
    const input = screen.getByLabelText(/type to confirm/i);
    input.focus();
    await userEvent.tab();
    expect(document.activeElement).toHaveTextContent("Cancel");
    await userEvent.tab();
    expect(document.activeElement).toHaveTextContent("Delete");
    await userEvent.tab();
    expect(document.activeElement).toBe(input);
    await userEvent.keyboard("{Escape}");
    expect(onClose).toHaveBeenCalled();
  });

  it("returns focus on close", async () => {
    const { getByText } = render(<Button>Trigger</Button>);
    const trigger = getByText("Trigger");
    const onClose = () => trigger.focus();
    render(
      <ConfirmDialog open onClose={onClose} onConfirm={() => {}} title="Delete" />
    );
    await userEvent.keyboard("{Escape}");
    expect(document.activeElement).toBe(trigger);
  });
});
