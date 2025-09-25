import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { axe } from "jest-axe";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "./Tabs";

describe("Tabs", () => {
  it("switches content on trigger click", async () => {
    const user = userEvent.setup();
    const { container } = render(
      <Tabs defaultValue="one">
        <TabsList>
          <TabsTrigger value="one">One</TabsTrigger>
          <TabsTrigger value="two">Two</TabsTrigger>
        </TabsList>
        <TabsContent value="one">First</TabsContent>
        <TabsContent value="two">Second</TabsContent>
      </Tabs>
    );
    const tab1 = screen.getByRole("tab", { name: "One" });
    const tab2 = screen.getByRole("tab", { name: "Two" });
    expect(tab1).toHaveAttribute("aria-selected", "true");
    expect(tab2).toHaveAttribute("aria-selected", "false");
    expect(screen.getByText("First")).toBeInTheDocument();
    expect(screen.queryByText("Second")).not.toBeInTheDocument();
    await user.click(tab2);
    expect(tab2).toHaveAttribute("aria-selected", "true");
    expect(screen.getByText("Second")).toBeInTheDocument();
    expect(screen.queryByText("First")).not.toBeInTheDocument();
    expect(await axe(container)).toHaveNoViolations();
  });

  it("allows arrow key navigation", async () => {
    const user = userEvent.setup();
    render(
      <Tabs defaultValue="one">
        <TabsList>
          <TabsTrigger value="one">One</TabsTrigger>
          <TabsTrigger value="two">Two</TabsTrigger>
          <TabsTrigger value="three">Three</TabsTrigger>
        </TabsList>
        <TabsContent value="one">First</TabsContent>
        <TabsContent value="two">Second</TabsContent>
        <TabsContent value="three">Third</TabsContent>
      </Tabs>
    );
    const [tab1, tab2, tab3] = screen.getAllByRole("tab");
    tab1.focus();
    await user.keyboard("{ArrowRight}");
    expect(tab2).toHaveFocus();
    expect(tab2).toHaveAttribute("aria-selected", "true");
    await user.keyboard("{ArrowRight}");
    expect(tab3).toHaveFocus();
    expect(tab3).toHaveAttribute("aria-selected", "true");
    await user.keyboard("{ArrowLeft}");
    expect(tab2).toHaveFocus();
    expect(tab2).toHaveAttribute("aria-selected", "true");
  });
});
