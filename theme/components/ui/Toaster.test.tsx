import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { axe } from "jest-axe";
import { Toaster, useToaster } from "./Toaster";
import { Button } from "./Button";

const Wrapper = () => {
  const { push } = useToaster();
  return <Button onClick={() => push("Saved")}>Notify</Button>;
};

describe("Toaster", () => {
  it("shows toast and closes with Escape, returning focus", async () => {
    const { container } = render(
      <Toaster>
        <Wrapper />
      </Toaster>
    );
    const trigger = screen.getByText("Notify");
    await userEvent.click(trigger);
    const toast = await screen.findByRole("status");
    expect(toast).toHaveTextContent("Saved");
    toast.focus();
    await userEvent.keyboard("{Escape}");
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
    expect(document.activeElement).toBe(trigger);
    expect(await axe(container)).toHaveNoViolations();
  });
});
