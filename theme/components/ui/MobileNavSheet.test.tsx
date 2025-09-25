import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { axe } from "jest-axe";
import { MobileNavSheet } from "./MobileNavSheet";
import { Button } from "./Button";
import { vi } from "vitest";

const links = [
  { label: "Home", href: "#home" },
  { label: "About", href: "#about" },
];

describe("MobileNavSheet", () => {
  it("traps focus and closes with Escape", async () => {
    const onClose = vi.fn();
    const { container } = render(
      <MobileNavSheet open onClose={onClose} links={links} />
    );
    const first = screen.getByText("Home");
    first.focus();
    await userEvent.tab();
    expect(document.activeElement).toHaveTextContent("About");
    await userEvent.tab();
    expect(document.activeElement).toBe(first);
    await userEvent.keyboard("{Escape}");
    expect(onClose).toHaveBeenCalled();
    expect(await axe(container)).toHaveNoViolations();
  });

  it("returns focus on close", async () => {
    const { getByText } = render(<Button>Trigger</Button>);
    const trigger = getByText("Trigger");
    const onClose = () => trigger.focus();
    render(
      <MobileNavSheet open onClose={onClose} links={links} />
    );
    await userEvent.keyboard("{Escape}");
    expect(document.activeElement).toBe(trigger);
  });
});
