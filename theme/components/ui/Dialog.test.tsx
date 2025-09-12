import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { axe } from "jest-axe";
import { Dialog } from "./Dialog";
import { Button } from "./Button";
import { vi } from "vitest";

describe("Dialog", () => {
  it("opens and closes with Escape", async () => {
    const onClose = vi.fn();
    const { container } = render(
      <Dialog open onClose={onClose}>
        <p>Content</p>
      </Dialog>
    );
    expect(screen.getByRole("dialog")).toBeInTheDocument();
    await userEvent.keyboard("{Escape}");
    expect(onClose).toHaveBeenCalled();
    expect(await axe(container)).toHaveNoViolations();
  });

  it("returns focus on close", async () => {
    const { getByText } = render(<Button>Trigger</Button>);
    const trigger = getByText("Trigger");
    const onClose = () => trigger.focus();
    render(
      <Dialog open onClose={onClose}>
        <Button onClick={onClose}>Close</Button>
      </Dialog>
    );
    await userEvent.click(screen.getByText("Close"));
    expect(document.activeElement).toBe(trigger);
  });
});
