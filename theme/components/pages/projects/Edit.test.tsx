import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { axe } from "jest-axe";
import { Edit } from "./Edit";

describe("Projects Edit", () => {
  it("supports keyboard navigation", async () => {
    render(<Edit />);
    const user = userEvent.setup();
    await user.tab();
    expect(screen.getByLabelText("Name")).toHaveFocus();
    await user.tab();
    expect(screen.getByLabelText("Description")).toHaveFocus();
    await user.tab();
    expect(screen.getByRole("button", { name: /cancel/i })).toHaveFocus();
    await user.tab();
    expect(screen.getByRole("button", { name: /save/i })).toHaveFocus();
  });

  it("shows validation errors", async () => {
    render(<Edit />);
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: /save/i }));
    expect(screen.getByText("Name is required")).toBeVisible();
    const input = screen.getByLabelText("Name");
    expect(input).toHaveAttribute("aria-describedby", expect.stringContaining("name-error"));
  });

  it("is accessible", async () => {
    const { container } = render(<Edit />);
    expect(await axe(container)).toHaveNoViolations();
  });
});
