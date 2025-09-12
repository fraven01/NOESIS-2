import { render, screen, waitFor } from "@testing-library/react";
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

  it("resets input when reopened", async () => {
    const { rerender } = render(
      <ConfirmDialog
        open
        onClose={() => {}}
        onConfirm={() => {}}
        title="Delete"
        confirmText="CONFIRM"
      />
    );
    const input = screen.getByLabelText(/type to confirm/i);
    const confirm = screen.getByRole("button", { name: "Delete" });
    await userEvent.type(input, "CONFIRM");
    expect(confirm).toBeEnabled();
    rerender(
      <ConfirmDialog
        open={false}
        onClose={() => {}}
        onConfirm={() => {}}
        title="Delete"
        confirmText="CONFIRM"
      />
    );
    await waitFor(() =>
      expect(screen.queryByLabelText(/type to confirm/i)).not.toBeInTheDocument()
    );
    rerender(
      <ConfirmDialog
        open
        onClose={() => {}}
        onConfirm={() => {}}
        title="Delete"
        confirmText="CONFIRM"
      />
    );
    expect(screen.getByLabelText(/type to confirm/i)).toHaveValue("");
    expect(screen.getByRole("button", { name: "Delete" })).toBeDisabled();
  });
});
