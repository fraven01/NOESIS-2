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
  it("restores focus after auto-hide when toast retains focus", async () => {
    const user = userEvent.setup();
    render(
      <Toaster>
        <Wrapper />
      </Toaster>
    );
    const trigger = screen.getByText("Notify");
    await user.click(trigger);
    const toast = screen.getByRole("status");
    toast.focus();
    await new Promise((r) => setTimeout(r, 3100));
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
    expect(document.activeElement).toBe(trigger);
  });

  it("does not restore focus when auto-hide occurs after focus moved", async () => {
    const user = userEvent.setup();
    const Extra = () => {
      const { push } = useToaster();
      return (
        <>
          <Button onClick={() => push("Saved")}>Notify</Button>
          <input aria-label="field" />
        </>
      );
    };
    render(
      <Toaster>
        <Extra />
      </Toaster>
    );
    const trigger = screen.getByText("Notify");
    const field = screen.getByLabelText("field");
    await user.click(trigger);
    screen.getByRole("status");
    await user.click(field);
    await new Promise((r) => setTimeout(r, 3100));
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
    expect(document.activeElement).toBe(field);
  });

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
