import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { axe } from "jest-axe";
import { Detail } from "./Detail";

describe("Detail", () => {
  it("navigates tabs via keyboard and is accessible", async () => {
    const user = userEvent.setup();
    const { container } = render(<Detail title="Alpha" status="Active" />);
    const tabs = screen.getAllByRole("tab");
    tabs[0].focus();
    await user.keyboard("{ArrowRight}");
    expect(tabs[1]).toHaveFocus();
    expect(screen.getByRole("heading", { name: "History" })).toBeInTheDocument();
    await user.keyboard("{ArrowRight}");
    expect(tabs[2]).toHaveFocus();
    expect(screen.getByText(/drag and drop/i)).toBeInTheDocument();
    expect(await axe(container)).toHaveNoViolations();
  });

  it("returns focus to trigger after dialog close", async () => {
    const user = userEvent.setup();
    render(<Detail title="Alpha" status="Active" />);
    const trigger = screen.getByRole("button", { name: "Edit" });
    await user.click(trigger);
    const close = screen.getByRole("button", { name: "Close" });
    await user.click(close);
    expect(trigger).toHaveFocus();
  });
});
