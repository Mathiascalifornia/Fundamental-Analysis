from tkinter import Tk, Label, Entry, ttk, Button


class MainWindows:

    def main(self, App):

        root = Tk()
        root.title("Analyse")
        label_language = Label(root, text="Language").grid(row=0, column=0)
        label_ticker = Label(
            root, text="Ticker de l'entreprise / ticker of the company"
        ).grid(row=1, column=0)
        label_name = Label(
            root, text="Nom de l'entreprise / name of the companie"
        ).grid(row=2, column=0)
        label_path = Label(root, text="Chemin de sauvegarde / save path").grid(
            row=3, column=0
        )
        label_button = Label(root, text="Débuter l'analyse / Start the analysis").grid(
            row=4, column=0
        )
        language = ttk.Combobox(root, values=["Français", "English"], width=22)
        language.grid(row=0, column=1)
        ticker = Entry(root, width=25, borderwidth=3)
        ticker.grid(row=1, column=1)
        name = Entry(root, width=25, borderwidth=3)
        name.grid(row=2, column=1)
        path = Entry(root, width=25, borderwidth=3)
        path.grid(row=3, column=1)

        def __get_entries(ticker=ticker, path=path):
            """Triggered by the analysis button. Get the entries and start the run function."""
            language_ = language.get()
            ticker_ = ticker.get()
            path_ = path.get()
            name_ = name.get()
            App(
                language=language_,
                ticker=ticker_,
                path_to_save=path_,
                company_name=name_,
            ).main()
            root.quit()

        button = Button(
            root,
            text="Analyse / Analysis",
            command=__get_entries,
            width=21,
            height=0,
            borderwidth=1,
        ).grid(row=4, column=1)
        root.mainloop()
