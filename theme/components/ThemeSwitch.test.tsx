import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { axe } from "jest-axe";
import { vi } from "vitest";
import { ThemeSwitch } from "./ThemeSwitch";

function stubMatchMedia(matches: boolean) {
  return vi.fn().mockImplementation(() => ({
    matches,
    media: "",
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  }));
}

describe("ThemeSwitch", () => {
  beforeEach(() => {
    window.matchMedia = stubMatchMedia(false);
    localStorage.clear();
    document.documentElement.className = "";
  });

  it("respects system preference", () => {
    window.matchMedia = stubMatchMedia(true);
    render(<ThemeSwitch />);
    expect(document.documentElement.classList.contains("dark")).toBe(true);
  });

  it("toggles theme and persists choice", async () => {
    const user = userEvent.setup();
    render(<ThemeSwitch />);
    const button = screen.getByRole("button", { name: /toggle theme/i });
    await user.click(button);
    expect(document.documentElement.classList.contains("dark")).toBe(true);
    expect(localStorage.getItem("theme")).toBe("dark");
    await user.click(button);
    expect(document.documentElement.classList.contains("dark")).toBe(false);
    expect(localStorage.getItem("theme")).toBe("light");
  });

  it("has no accessibility violations", async () => {
    const { container } = render(<ThemeSwitch />);
    const results = await axe(container);
    expect(results).toHaveNoViolations();
  });
});
