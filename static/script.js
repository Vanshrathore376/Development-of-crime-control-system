const personForm =
    document.getElementById("personForm");

const personSelect =
    document.getElementById("personSelect");

const identifyButton =
    document.getElementById("identifyButton");

const personMessage =
    document.getElementById("personMessage");

const resultMessage =
    document.getElementById("resultMessage");

const resultSection =
    document.getElementById("resultSection");

const resultImage =
    document.getElementById("resultImage");

const resultSummary =
    document.getElementById("resultSummary");

const faceResults =
    document.getElementById("faceResults");


// ------------------------------------
// Load database persons
// ------------------------------------

async function loadPersons() {

    try {

        const response =
            await fetch("/api/persons");

        const persons =
            await response.json();

        personSelect.innerHTML =
            '<option value="">Select person</option>';

        persons.forEach(person => {

            const option =
                document.createElement("option");

            option.value =
                person.id;

            option.textContent =
                `${person.name} (${person.case_id || "No Case ID"})`;

            personSelect.appendChild(option);

        });

    } catch (error) {

        console.error(error);

    }
}


// ------------------------------------
// Register person
// ------------------------------------

personForm.addEventListener(
    "submit",
    async function(event) {

        event.preventDefault();

        personMessage.textContent =
            "Processing reference image...";

        const formData =
            new FormData();

        formData.append(
            "name",
            document.getElementById(
                "personName"
            ).value
        );

        formData.append(
            "case_id",
            document.getElementById(
                "caseId"
            ).value
        );

        formData.append(
            "image",
            document.getElementById(
                "referenceImage"
            ).files[0]
        );

        try {

            const response =
                await fetch(
                    "/api/person/add",
                    {
                        method: "POST",
                        body: formData
                    }
                );

            const data =
                await response.json();

            if (data.success) {

                personMessage.textContent =
                    `✓ ${data.message}`;

                personForm.reset();

                await loadPersons();

            } else {

                personMessage.textContent =
                    `✗ ${data.message}`;
            }

        } catch (error) {

            personMessage.textContent =
                "Server error.";

            console.error(error);
        }
    }
);


// ------------------------------------
// Identify person
// ------------------------------------

identifyButton.addEventListener(
    "click",
    async function() {

        const personId =
            personSelect.value;

        const image =
            document.getElementById(
                "groupImage"
            ).files[0];

        if (!personId) {

            resultMessage.textContent =
                "Please select a person.";

            return;
        }

        if (!image) {

            resultMessage.textContent =
                "Please upload a group image.";

            return;
        }

        resultMessage.textContent =
            "Analyzing image...";

        resultSection.style.display =
            "block";

        const formData =
            new FormData();

        formData.append(
            "person_id",
            personId
        );

        formData.append(
            "group_image",
            image
        );

        try {

            const response =
                await fetch(
                    "/api/identify",
                    {
                        method: "POST",
                        body: formData
                    }
                );

            const data =
                await response.json();

            if (!data.success) {

                resultMessage.textContent =
                    data.message;

                return;
            }

            resultMessage.textContent =
                "";

            resultImage.src =
                data.result_image;

            resultSummary.innerHTML = `
                <p>
                    <strong>Person:</strong>
                    ${data.person.name}
                </p>

                <p>
                    <strong>Case ID:</strong>
                    ${data.person.case_id || "N/A"}
                </p>

                <p>
                    <strong>Faces detected:</strong>
                    ${data.faces_detected}
                </p>

                <p>
                    <strong>Match found:</strong>
                    ${data.matched ? "YES" : "NO"}
                </p>

                ${
                    data.matched
                    ?
                    `<p>
                        <strong>Best similarity:</strong>
                        ${data.best_similarity}
                    </p>`
                    :
                    ""
                }
            `;

            faceResults.innerHTML =
                "";

            data.results.forEach(
                result => {

                    const div =
                        document.createElement(
                            "div"
                        );

                    div.className =
                        result.matched
                        ? "match"
                        : "no-match";

                    div.innerHTML = `
                        <strong>
                            Face ${result.face_index}
                        </strong>

                        <br>

                        Similarity:
                        ${result.similarity}

                        <br>

                        Euclidean/L2 Distance:
                        ${result.distance}

                        <br>

                        Result:
                        ${
                            result.matched
                            ? "MATCH"
                            : "NO MATCH"
                        }
                    `;

                    faceResults.appendChild(
                        div
                    );
                }
            );

        } catch (error) {

            resultMessage.textContent =
                "Unable to process image.";

            console.error(error);
        }
    }
);


// Initial load
loadPersons();